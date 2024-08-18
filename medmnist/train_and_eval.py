import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy
from models import ResNet18, ResNet50, VisionTransformer, VisionTransformerHuggingFace, VisionTransformerTimm, MedCLIPViTModel, ResNet3D18, ViT3D
from utility import get_datasets, get_dataloaders
from medmnist import Evaluator
from medmnist.evaluator import getAUC, getACC 
from tqdm import tqdm, trange
import wandb

# --- Checkpoint functions ---
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), "best_" + os.path.basename(filename))
        torch.save(state, best_filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_auc = checkpoint['best_auc']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return model, optimizer, start_epoch, best_auc
    else:
        print(f"No checkpoint found at '{filename}'")
        return model, optimizer, 0, 0

# --- Training function ---
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Ensure targets are 1D
        if targets.dim() > 1:
            targets = targets.squeeze()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    accuracy = correct / total
    return total_loss / len(train_loader.dataset), accuracy

# --- Evaluation function ---
def evaluate(model, dataloader, criterion, evaluator, device, save_folder, run, epoch=None):
    model.eval()
    total_loss = 0
    y_score = []
    y_true = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            # Ensure targets are 1D
            if targets.dim() > 1:
                targets = targets.squeeze()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            y_score.append(outputs.cpu())
            y_true.append(targets.cpu())

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
    y_score = torch.cat(y_score).numpy()
    y_true = torch.cat(y_true).numpy()
    auc = getAUC(y_true, y_score, evaluator.info['task'])
    acc = correct / total

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

# --- Main function ---
def main(args):
    wandb.login(key="2034da31c29a117a10e74550ff9896c178344596", relogin=True)

    # Create a custom config dictionary
    config = {
        "dataset": args.data_flag,
        "model": args.model_flag,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        # "pretrained": args.pretrained,
        "resize": args.resize
    }
    wandb_run_name = f"{args.data_flag}, {args.model_flag}, epochs {args.num_epochs}, BS {args.batch_size}, LR {args.lr}"
    # Set the W&B run mode based on whether it's a resume or a new run
    resume_run = False
    if args.model_path:
        resume_run = True
    # Initialize W&B with the custom config
    if resume_run:
        wandb_id = wandb.util.generate_id()  # Generate a unique ID
        wandb.init(project="medMnist-experiments", 
                entity="rozadler-rd-university-of-surrey", 
                config=config, 
                name=wandb_run_name,
                id=wandb_id,
                resume="allow",  # This allows resuming
                settings=wandb.Settings(symlink=False)
                )
    else:
        wandb.init(project="medMnist-experiments", 
                entity="rozadler-rd-university-of-surrey", 
                config=config, 
                name=wandb_run_name,
                settings=wandb.Settings(symlink=False)
                )
    # wandb.init(project="medMnist-experiments", 
    #            entity="rozadler-rd-university-of-surrey", 
    #            config=config, 
    #            name=wandb_run_name,
    #            resume=resume_run,  # Resuming W&B run if checkpoint exists
    #            settings=wandb.Settings(symlink=False)
    #            )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if torch.cuda.is_available():
        device = torch.device('cuda') 
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets and dataloaders (now with size and shape_transform parameters)
    train_dataset, val_dataset, test_dataset = get_datasets(
        args.data_flag, args.download, args.as_rgb, args.resize, 
        args.model_flag, size=args.size, shape_transform=args.shape_transform
    )
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size, num_workers=4)

    n_channels = 3 if args.as_rgb else train_dataset.info['n_channels']
    num_classes = len(train_dataset.info['label'])

     # Initialize model
    if args.model_flag == 'resnet18':
        # if "3d" in args.data_flag:
        #     model = ResNet3D18(n_channels, num_classes)
        #     model = convert_to_acs_or_conv3d(model, conv_type=args.conv_type)
        # else:
        model = ResNet18(n_channels, num_classes)
    elif args.model_flag == 'vit_timm_3d':
        model = ViT3D(num_classes, pretrained=args.pretrained)
    elif args.model_flag == 'resnet50':
        model = ResNet50(n_channels, num_classes)
    elif args.model_flag == 'vit':
        model = VisionTransformer(num_classes)
    elif args.model_flag == 'vit_hf':
        model = VisionTransformerHuggingFace(num_classes)
    elif args.model_flag == 'vit_timm':
        model = VisionTransformerTimm(num_classes, pretrained=args.pretrained)
    elif args.model_flag == 'medclip_vit':
        model = MedCLIPViTModel(num_classes)
    else:
        raise ValueError("Unknown model flag")
    
    # Support multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # --- Checkpointing: Load checkpoint if fine-tuning ---
    start_epoch = 0
    best_auc = 0
    if args.fine_tune and args.model_path:
        model, optimizer, start_epoch, best_auc = load_checkpoint(model, optimizer, filename=args.model_path)
        print(f"checkpoint here .")
    train_evaluator = Evaluator(args.data_flag, 'train')
    val_evaluator = Evaluator(args.data_flag, 'val')
    test_evaluator = Evaluator(args.data_flag, 'test')

    
    best_model = deepcopy(model)
    early_stopping_patience = 40  # or any other value
    patience_counter = 0

# --- Training loop ---
    for epoch in trange(start_epoch, args.num_epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, val_evaluator, device, save_folder=args.output_dir, run=f'epoch_{epoch+1}', epoch=epoch)

            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "epoch": epoch+1})

            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc
                best_model = deepcopy(model)
                torch.save(best_model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=os.path.join(args.output_dir, 'checkpoint.pth.tar'))

            scheduler.step()

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    # for epoch in trange(start_epoch, args.num_epochs):
    #     train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)  # Unpack the returned tuple
    #     val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, val_evaluator, device, save_folder=args.output_dir, run=f'epoch_{epoch+1}', epoch=epoch)
        
    #     # Log training metrics to W&B
    #     wandb.log({"train_loss": train_loss, "train_acc": train_acc, "epoch": epoch+1})  # Log train accuracy as well

    #     Determine if this is the best model so far
    #     is_best = val_auc > best_auc
    #     if is_best:
    #         best_auc = val_auc
    #         best_model = deepcopy(model)
    #         # Save the best model
    #         model_save_path = os.path.join(args.output_dir, 'best_model.pth')
    #         torch.save(best_model.state_dict(), model_save_path)
        
    #     # --- Save the current checkpoint ---
    #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'best_auc': best_auc,
    #         'optimizer': optimizer.state_dict(),
    #     }, is_best, filename=checkpoint_path)
        
    #     scheduler.step()
     # Now the print statement correctly formats the values
    print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}')

    # --- Final evaluation on test set ---
    test_loss, test_auc, test_acc = evaluate(best_model, test_loader, criterion, test_evaluator, device, save_folder=args.output_dir, run='test')
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}')

    # Final log for test results
    wandb.log({"test_loss": test_loss, "test_auc": test_auc, "test_acc": test_acc})

    # Finish the run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_flag', type=str, required=True)
    parser.add_argument('--model_flag', type=str, required=True, choices=['resnet18', 'resnet50', 'vit', 'vit_hf', 'vit_timm', 'medclip_vit', 'vit_timm_3d'])
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
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 75])  # Default milestones 
    parser.add_argument('--size', type=int, default=28, help='Size of the 3D dataset voxels (e.g., 28 or 64)')
    parser.add_argument('--shape_transform', action='store_true', help='Apply shape-specific transformations for 3D data')
    parser.add_argument('--conv_type', type=str, default='ACSConv', choices=['ACSConv', 'Conv2_5d', 'Conv3d'], help='Type of convolution for 3D models')
    args = parser.parse_args()
    
    main(args)