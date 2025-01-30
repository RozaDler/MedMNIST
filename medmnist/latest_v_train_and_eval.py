import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import timm
from copy import deepcopy
from models import ResNet18, ResNet50, VisionTransformerTimm, MedCLIPViTModel, ResNet3D18, ViT3D, ResNet3D_ViT3D
from utility import get_datasets, get_dataloaders
from medmnist import Evaluator
from medmnist.evaluator import getAUC, getACC 
from tqdm import tqdm, trange
import wandb
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from timm.models.vision_transformer import vit_base_patch16_224
from timm import create_model
from models import convert_to_acs_or_conv3d
import random


def load_pretrained_vit_weights(vit3d_model, vit2d_model):
    # Load transformer encoder weights layer by layer
    for i, layer in enumerate(vit3d_model.transformer.layers):
        vit3d_model.transformer.layers[i].norm1.weight.data = vit2d_model.blocks[i].norm1.weight.data
        vit3d_model.transformer.layers[i].norm1.bias.data = vit2d_model.blocks[i].norm1.bias.data
        vit3d_model.transformer.layers[i].norm2.weight.data = vit2d_model.blocks[i].norm2.weight.data
        vit3d_model.transformer.layers[i].norm2.bias.data = vit2d_model.blocks[i].norm2.bias.data
        vit3d_model.transformer.layers[i].self_attn.in_proj_weight.data = vit2d_model.blocks[i].attn.qkv.weight.data
        vit3d_model.transformer.layers[i].self_attn.in_proj_bias.data = vit2d_model.blocks[i].attn.qkv.bias.data
        vit3d_model.transformer.layers[i].self_attn.out_proj.weight.data = vit2d_model.blocks[i].attn.proj.weight.data
        vit3d_model.transformer.layers[i].self_attn.out_proj.bias.data = vit2d_model.blocks[i].attn.proj.bias.data
        vit3d_model.transformer.layers[i].linear1.weight.data = vit2d_model.blocks[i].mlp.fc1.weight.data
        vit3d_model.transformer.layers[i].linear1.bias.data = vit2d_model.blocks[i].mlp.fc1.bias.data
        vit3d_model.transformer.layers[i].linear2.weight.data = vit2d_model.blocks[i].mlp.fc2.weight.data
        vit3d_model.transformer.layers[i].linear2.bias.data = vit2d_model.blocks[i].mlp.fc2.bias.data

    # Adjust the positional embeddings
    vit3d_pos_embed = vit3d_model.embedding.pos_embed  # This should be (1, num_patches + 1, embedding_dim)
    vit2d_pos_embed = vit2d_model.pos_embed[:, 1:, :]  # Skip the class token in 2D model

    # Interpolate to match the number of patches
    vit3d_pos_embed_new = F.interpolate(vit2d_pos_embed.unsqueeze(0), size=(vit3d_model.num_patches, vit3d_pos_embed.shape[-1]), mode='bilinear', align_corners=False)
    vit3d_pos_embed_new = vit3d_pos_embed_new.squeeze(0)  # Remove the batch dimension

    # Combine the class token with the interpolated positional embeddings
    combined_pos_embed = torch.cat([vit3d_pos_embed[:, :1, :], vit3d_pos_embed_new], dim=1)
    
    # Update the entire positional embedding tensor
    vit3d_model.embedding.pos_embed = nn.Parameter(combined_pos_embed)

    return vit3d_model

class OversampledDataset(Dataset):
    def __init__(self, original_dataset, minority_class_label, oversample_factor=1, transform=None):
        self.original_dataset = original_dataset
        self.minority_class_label = minority_class_label
        self.oversample_factor = oversample_factor
        self.indices = self._get_indices()

    def _get_indices(self):
        minority_indices = [i for i, (_, label) in enumerate(self.original_dataset) if label == self.minority_class_label]
        oversampled_indices = minority_indices * self.oversample_factor
        return list(range(len(self.original_dataset))) + oversampled_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]
    
#real train function uncomment after one class
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 3D RESNET we add this multiplication to match medmnist 
         # Apply random multiplication to the input as part of regularization
        # inputs = inputs * torch.rand(1).item()

         # Ensure targets are 1D
        if targets.dim() > 1:
            targets = targets.squeeze()

        optimizer.zero_grad()
        outputs = model(inputs)

        # # Print the outputs for the first few batches this output logging during each epoch
        # print(f"Epoch output example: {outputs[:5]}")

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(train_loader.dataset)


# real evaluate function uncomment after one class
def evaluate(model, dataloader, criterion, evaluator, device, save_folder, run, epoch=None):
    model.eval()
    total_loss = 0
    y_score = []
    y_true = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 3D RESNET - Multiply inputs by a fixed coefficient during evaluation- to follow medmnist for 3d 
            # inputs = inputs * 0.5

            # Ensure targets are 1D
            if targets.dim() > 1:
                targets = targets.squeeze()

            outputs = model(inputs)
            print(f"Output shape: {outputs.shape}")  # This should print (batch_size, 2) we are ensuring the model is outputting logit with correct shape
            #should see output shapes like (32, 2) (assuming a batch size of 32). If the output shape is correct, then your model is outputting logits, and you can safely proceed with the softmax step during evaluation.
            print(f"Logits: {outputs[:5]}")  # Print the first 5 logits to check their values
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            y_score.append(outputs.cpu())
            y_true.append(targets.cpu())
    y_score = torch.cat(y_score).numpy()
    y_true = torch.cat(y_true).numpy()
    auc = getAUC(y_true, y_score, evaluator.info['task'])
    acc = getACC(y_true, y_score, evaluator.info['task'])

    # Log metrics to W&B
    wandb.log({"val_loss": total_loss / len(dataloader.dataset), "val_auc": auc, "val_acc": acc, "epoch": epoch+1 if epoch else 0})

    # Save evaluation results if save_folder is specified
    if save_folder:
        if not run:
            run = 'evaluation'
        result_file = os.path.join(save_folder, f'{run}_results.txt')
        with open(result_file, 'w') as f:
            f.write(f'AUC: {auc}\n')
            f.write(f'Accuracy: {acc}\n')

    return total_loss / len(dataloader.dataset), auc, acc

def main(args):
    wandb.login(key="2034da31c29a117a10e74550ff9896c178344596", relogin=True)

    # Create a custom config dictionary
    config = {
        "dataset": args.data_flag,
        "model": args.model_flag,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "pretrained": args.pretrained,
        "resize": args.resize
    }
    wandb_run_name = f"{args.data_flag}, non-pretrained {args.model_flag}, epochs {args.num_epochs}, BS {args.batch_size}, LR {args.lr}"
        # Initialize W&B with the custom config
    wandb.init(project="medMnist-experiments", 
               entity="rozadler-rd-university-of-surrey", 
               config=config, 
               name=wandb_run_name,
               settings=wandb.Settings(symlink=False)
               ) #potentially add settings=wandb.Settings(symlink=False)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # train_dataset, val_dataset, test_dataset = get_datasets(args.data_flag, args.download, args.as_rgb, args.resize)   commented for change in medclip 
    # train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size)

    # Load datasets
    train_dataset, val_dataset, test_dataset = get_datasets(args.data_flag, args.download, args.as_rgb, args.resize, args.model_flag)

    # Initialize counts for each class
    class_counts = [0] * len(train_dataset.info['label'])
    for _, label in train_dataset:
        if isinstance(label, torch.Tensor):  # If label is a tensor, convert to an int
            label = label.item()
        label = int(label)  # Ensure the label is an integer
        class_counts[label] += 1

    # Print class distribution
    print(f"Class distribution: {class_counts}")
    
    #  ------ method 2 manual random oversampling-----------
    minority_class_label = 1  # Assuming the minority class label is 1
    oversample_factor = 4     # Number of times to oversample the minority class

    # oversampled_train_dataset = OversampledDataset(train_dataset, minority_class_label, oversample_factor)
    # train_loader = DataLoader(oversampled_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # After creating the oversampled dataset
    # oversampled_class_counts = [0] * len(train_dataset.info['label'])
    # for _, label in oversampled_train_dataset:
    #     if isinstance(label, torch.Tensor):
    #         label = label.item()  # Convert tensor to an integer
    #     label = int(label)  # Ensure label is an integer
    #     oversampled_class_counts[label] += 1

    # print(f"Class distribution after oversampling: {oversampled_class_counts}")
        # ------ end of method 2 manual random oversampling-----------

    # # Create DataLoader for 2d or 3d without oversampling 
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size, num_workers=4)

    n_channels = 3 if args.as_rgb else train_dataset.info['n_channels']
    num_classes = len(train_dataset.info['label'])

    if args.model_flag == 'resnet18':
        if "3d" in args.data_flag:
            model = ResNet3D18(n_channels, num_classes)
            model = convert_to_acs_or_conv3d(model, conv_type=args.conv_type)
        else:
            model = ResNet18(n_channels, num_classes)
    elif args.model_flag == 'vit_3d':
        model = ViT3D(img_dim=28, patch_size=7, num_classes=2, embedding_dim=768, depth=6, heads=8, mlp_dim=512, channels=n_channels)
        if args.pretrained:
            # Load the pretrained 2D ViT model
            vit2d_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            
            # Load pretrained weights into the 3D model
            model = load_pretrained_vit_weights(model, vit2d_model)

    elif args.model_flag == 'resnet50':
        model = ResNet50(n_channels, num_classes)
    elif args.model_flag == 'vit':
        model = VisionTransformer(num_classes)
    elif args.model_flag == 'vit_timm':
        model = VisionTransformerTimm(num_classes, pretrained=args.pretrained)
    elif args.model_flag == 'medclip_vit':
        model = MedCLIPViTModel(num_classes)
    elif args.model_flag == 'resnet3d_vit':
        model = ResNet3D_ViT3D(resnet_channels=n_channels, vit_channels=n_channels, num_classes=num_classes)
        convert_to_acs_or_conv3d(model.resnet, conv_type=args.conv_type)
    elif args.model_flag == 'resnet_vit_3d_extractor':
        model = ResNet3D_ViT3D_Extractor(resnet_channels=n_channels, vit_channels=n_channels, num_classes=num_classes)
        convert_to_acs_or_conv3d(model.resnet, conv_type='Conv3d')  # Use 3D convolutions throughout
    else:
        raise ValueError("Unknown model flag")

    model = model.to(device)

    #Check the model’s initial outputs before any training (i.e., before the first epoch)
    # After model instantiation and data loader creation
    with torch.no_grad():
        model.eval()
        sample_input, _ = next(iter(train_loader))
        sample_input = sample_input.to(device)
        initial_output = model(sample_input)
        print(f"Initial output: {initial_output}")

    # After model instantiation in main
    # print_model_summary(model, (n_channels, 28, 28, 28))  # Assuming your input size is (1, 28, 28, 28) for 3D single-channel data
    # Use the calculated class weights in the loss function
    # criterion = nn.CrossEntropyLoss(weight=class_weights) # to help 3d
    criterion = nn.CrossEntropyLoss() # uncomment if you dont want weighted loss (for 3d)
    # criterion = nn.BCEWithLogitsLoss()  # Using binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    scheduler = MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1) #3D scheduler for resnet 18 3d
    early_stopping_patience = 15  # Stop if no improvement for 15 epochs 3D

    if args.fine_tune and args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    train_evaluator = Evaluator(args.data_flag, 'train')
    val_evaluator = Evaluator(args.data_flag, 'val')
    test_evaluator = Evaluator(args.data_flag, 'test')

    best_model = deepcopy(model)
    best_auc = 0
    epochs_without_improvement = 0

    for epoch in trange(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, val_evaluator, device, save_folder=args.output_dir, run=f'epoch_{epoch+1}', epoch=epoch)
        # Log training metrics to W&B
        wandb.log({"train_loss": train_loss, "epoch": epoch+1})
        
        #add this back for 2d or everything else 
        # if val_auc > best_auc:
        #     best_auc = val_auc
        #     best_model = deepcopy(model)
        #     # Save the best model
        #     model_save_path = os.path.join(args.output_dir, 'best_model.pth')
        #     torch.save(best_model.state_dict(), model_save_path)
        
        # use this instead for resnet18 3D
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = deepcopy(model)
            epochs_without_improvement = 0
            # Save the best model
            model_save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(best_model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping due to no improvement in validation AUC")
            break

        scheduler.step()
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}')

    # Evaluate on test set with the best model
    test_loss, test_auc, test_acc = evaluate(best_model, test_loader, criterion, test_evaluator, device, save_folder=args.output_dir, run='test')
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}')

    # Final log for test results
    wandb.log({"test_loss": test_loss, "test_auc": test_auc, "test_acc": test_acc})
    # Finish the run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_flag', type=str, required=True)
    parser.add_argument('--model_flag', type=str, required=True, choices=['resnet18', 'resnet50', 'vit', 'vit_timm', 'medclip_vit', 'vit_3d', 'resnet3d_vit', 'resnet_vit_3d_extractor'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--as_rgb', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--fine_tune', action='store_true', help='Flag to indicate if the model should be fine-tuned')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--milestones', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation for 3D datasets')
    parser.add_argument('--conv_type', type=str, default='ACSConv', choices=['ACSConv', 'Conv2_5d', 'Conv3d'],
                        help='Type of convolution to use in the model')
    args = parser.parse_args()
    
    main(args)
