# inference.py

import argparse
import torch
from models import ResNet18, ResNet50, VisionTransformer, VisionTransformerHuggingFace, VisionTransformerTimm
from utility import get_datasets, get_dataloaders
from medmnist import Evaluator
from medmnist.evaluator import getAUC, getACC 
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, criterion, evaluator, device): #used to have  save_folder=None, run=None as parameters as well 
    model.eval()
    total_loss = 0
    y_score = []
    y_true = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.dim() > 1:
                targets = targets.squeeze()  # Ensure targets are 1D
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            y_score.append(outputs.cpu())
            y_true.append(targets.cpu())
    y_score = torch.cat(y_score).numpy()
    y_true = torch.cat(y_true).numpy()
    auc = getAUC(y_true, y_score, evaluator.info['task'])
    acc = getACC(y_true, y_score, evaluator.info['task'])
    return total_loss / len(dataloader.dataset), auc, acc

def main(args):
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, test_dataset = get_datasets(args.data_flag, args.download, args.as_rgb, args.resize)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size)

    n_channels = 3 if args.as_rgb else train_dataset.info['n_channels']
    if args.model_flag == 'resnet18':
        model = ResNet18(n_channels, len(train_dataset.info['label']), pretrained=args.pretrained)
    elif args.model_flag == 'resnet50':
        model = ResNet50(n_channels, len(train_dataset.info['label']), pretrained=args.pretrained)
    elif args.model_flag == 'vit':
        model = VisionTransformer(len(train_dataset.info['label']), pretrained=args.pretrained)
    # elif args.model_flag == 'vit_hf':
    #     model = VisionTransformerHuggingFace(len(train_dataset.info['label']))
    elif args.model_flag == 'vit_timm':
        model = VisionTransformerTimm(len(train_dataset.info['label']), pretrained=args.pretrained)
    elif args.model_flag == 'vit_hf':
        model = VisionTransformerHuggingFace(len(train_dataset.info['label']), pretrained=args.pretrained)
    else:
        raise ValueError("Unknown model flag")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    test_evaluator = Evaluator(args.data_flag, 'test')
    test_loss, test_auc, test_acc = evaluate(model, test_loader, criterion, test_evaluator, device)
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}')

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
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()
    
    main(args)
