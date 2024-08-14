import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy
from models import ResNet18, ResNet50, VisionTransformer, VisionTransformerHuggingFace, VisionTransformerTimm
from utility import get_datasets, get_dataloaders
from medmnist import Evaluator
from medmnist.evaluator import getAUC, getACC 
from tqdm import tqdm, trange
import wandb

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
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
    return total_loss / len(train_loader.dataset)

# Updated evaluate function without default values for save_folder and run
def evaluate(model, dataloader, criterion, evaluator, device, save_folder, run, epoch=None):
    model.eval()
    total_loss = 0
    y_score = []
    y_true = []
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
    wandb_run_name = f"{args.data_flag}, {args.model_flag}, epochs {args.num_epochs}, BS {args.batch_size}, LR {args.lr}"
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

    train_dataset, val_dataset, test_dataset = get_datasets(args.data_flag, args.download, args.as_rgb, args.resize)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size)

    n_channels = 3 if args.as_rgb else train_dataset.info['n_channels']
    num_classes = len(train_dataset.info['label'])

    if args.model_flag == 'resnet18':
        model = ResNet18(n_channels, num_classes)
    elif args.model_flag == 'resnet50':
        model = ResNet50(n_channels, num_classes)
    elif args.model_flag == 'vit':
        model = VisionTransformer(num_classes)
    elif args.model_flag == 'vit_hf':
        model = VisionTransformerHuggingFace(num_classes)
    elif args.model_flag == 'vit_timm':
        model = VisionTransformerTimm(num_classes, pretrained=args.pretrained)
    else:
        raise ValueError("Unknown model flag")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    if args.fine_tune and args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    train_evaluator = Evaluator(args.data_flag, 'train')
    val_evaluator = Evaluator(args.data_flag, 'val')
    test_evaluator = Evaluator(args.data_flag, 'test')

    best_model = deepcopy(model)
    best_auc = 0

    for epoch in trange(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, val_evaluator, device, save_folder=args.output_dir, run=f'epoch_{epoch+1}', epoch=epoch)
        
        # Log training metrics to W&B
        wandb.log({"train_loss": train_loss, "epoch": epoch+1})

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = deepcopy(model)
            # Save the best model
            model_save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(best_model.state_dict(), model_save_path)
        
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
    parser.add_argument('--model_flag', type=str, required=True, choices=['resnet18', 'resnet50', 'vit', 'vit_hf', 'vit_timm'])
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
    args = parser.parse_args()
    
    main(args)