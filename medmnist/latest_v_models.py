import torch.nn as nn 
from torchvision.models import resnet18, resnet50, vit_b_16, ViT_B_16_Weights
from torchvision.models.resnet import BasicBlock, ResNet
from transformers import ViTModel, ViTFeatureExtractor
import torch
from timm import create_model
from einops import rearrange, repeat
import logging
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel  
from medclip import MedCLIPProcessor
import requests
import zipfile
import os
from emb import Embedding
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter  # Import conversion classes
from torch.nn import BatchNorm2d as SynchronizedBatchNorm2d
from torch.nn import BatchNorm3d as SynchronizedBatchNorm3d
import torch.nn.functional as F # importing thsi for interpolation- need this in vit3d


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False 
        )
        self.model.fc = nn.Linear(
            self.model.fc.in_features, num_classes
        )
    def forward(self, x):
        return self.model(x)
    
class ResNet50(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
#this resnet3d works great for resnet3d18 alone but we wont use taht for the one with vit
class ResNet3D18(nn.Module):  
    def __init__(self, in_channels, num_classes):
        super(ResNet3D18, self).__init__()
        # Load the pre-defined resnet18 model
        self.model = resnet18(pretrained=False)

        # Modify the first convolutional layer to work with 3D inputs
        self.model.conv1 = nn.Conv3d(
            in_channels, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace all BatchNorm2d layers with BatchNorm3d
        self.model.bn1 = nn.BatchNorm3d(64)
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for block in getattr(self.model, layer_name):
                if hasattr(block, 'bn1'):
                    block.bn1 = nn.BatchNorm3d(block.bn1.num_features)
                if hasattr(block, 'bn2'):
                    block.bn2 = nn.BatchNorm3d(block.bn2.num_features)
                if hasattr(block, 'bn3'):  # For Bottleneck blocks
                    block.bn3 = nn.BatchNorm3d(block.bn3.num_features)

        # Modify the fully connected layer to match the number of output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to convert models to ACSConv, Conv2_5d, or Conv3d
def convert_to_acs_or_conv3d(model, conv_type='ACSConv'):
    if conv_type == 'ACSConv':
        model = ACSConverter(model)
    elif conv_type == 'Conv2_5d':
        model = Conv2_5dConverter(model)
    elif conv_type == 'Conv3d':
        model = Conv3dConverter(model)
    return model

#this just concatinates
class ResNet3D_ViT3D(nn.Module):
    def __init__(self, resnet_channels, vit_channels, num_classes):
        super(ResNet3D_ViT3D, self).__init__()
        self.resnet = ResNet3D18(in_channels=resnet_channels, num_classes=num_classes)
        self.vit = ViT3D(img_dim=28, patch_size=7, num_classes=num_classes, embedding_dim=768, depth=6, heads=8, mlp_dim=512, channels=vit_channels)
        
        # Adjust the final layers
        self.fc = nn.Linear(num_classes * 2, num_classes)
    
    def forward(self, x):
        resnet_features = self.resnet(x)
        vit_features = self.vit(x)
        
        # Concatenate or sum the features
        combined_features = torch.cat((resnet_features, vit_features), dim=1)
        out = self.fc(combined_features)
        
        return out
#for pretrained
# class VisionTransformerTimm(nn.Module):
#     def __init__(self, num_classes, pretrained):
#         super(VisionTransformerTimm, self).__init__()
#         self.vit = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
#         self.vit.head = nn.Linear(self.vit.head.in_features, num_classes) #added this head for fine tuning 

#     def forward(self, x):
#         return self.vit(x)

# for non-pretrained
from timm import create_model

class VisionTransformerTimm(nn.Module):
    def __init__(self, num_classes, pretrained=False):  # Default to non-pretrained
        super(VisionTransformerTimm, self).__init__()
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)  # Custom head for your dataset

    def forward(self, x):
        return self.vit(x)

#this works for non pretrained
class ViT3D(nn.Module):
    def __init__(self, img_dim=28, patch_size=7, num_classes=2, embedding_dim=256, depth=6, heads=8, mlp_dim=512, channels=3):
        super(ViT3D, self).__init__()
        self.embedding = Embedding(
            image_size=img_dim,
            patch_size=patch_size,
            num_patches=(img_dim // patch_size) ** 3,
            channels=channels,
            embedding_dim=embedding_dim,
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)  # Obtain patch embeddings and add positional embeddings
        x = self.transformer(x)  # Transformer encoder
        x = x.mean(dim=1)  # Global average pooling (alternatively, use the [CLS] token)
        x = self.mlp_head(x)  # Final classification head
        return x

#for pretrained 
# class ViT3D(nn.Module):
#     def __init__(self, img_dim=28, patch_size=7, num_classes=2, embedding_dim=768, depth=6, heads=8, mlp_dim=2048, channels=3):
#         super(ViT3D, self).__init__()
#         self.embedding = Embedding(
#             image_size=28,
#             patch_size=7,
#             num_patches=(img_dim // patch_size) ** 3,
#             channels=channels,
#             embedding_dim=embedding_dim,
#         )
#         self.num_patches = (img_dim // patch_size) ** 3

#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads, dim_feedforward=mlp_dim),
#             num_layers=depth
#         )

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(embedding_dim),
#             nn.Linear(embedding_dim, num_classes)
#         )

#     def forward(self, x):
#         x = self.embedding(x)  # Obtain patch embeddings and add positional embeddings
#         x = self.transformer(x)  # Transformer encoder
#         x = x.mean(dim=1)  # Global average pooling (alternatively, use the [CLS] token)
#         x = self.mlp_head(x)  # Final classification head
#         return x
# class VisionTransformerHuggingFace(nn.Module):
#     def __init__(self, num_classes, pretrained=False):
#         super(VisionTransformerHuggingFace, self).__init__()
#         if pretrained:
#             logger.info("Loading pretrained Hugging Face ViT model")
#             self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#         else:
#             logger.info("Loading non-pretrained Hugging Face ViT model")
#             self.vit = ViTModel(config=ViTModel.config_class())
        
#         self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

#     def forward(self, x):
#         outputs = self.vit(pixel_values=x).last_hidden_state
#         logits = self.classifier(outputs[:, 0, :])
#         return logits

class MedCLIPViTModel(nn.Module):
    def __init__(self, num_classes):
        super(MedCLIPViTModel, self).__init__()
        # Initialize the model
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

        # Loading the pretrained weights manually
        self.load_pretrained_vit_weights()

        # Adding a final classification layer based on MedCLIP's architecture
        self.classifier = nn.Linear(512, num_classes)  # 512 comes from the projection head output size

    def load_pretrained_vit_weights(self):
        # URL for the ViT weights
        vit_weights_url = "https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip"
        vit_weights_path = "./medclip_vit_weights.zip"

        # Download the weights if they don't already exist locally
        if not os.path.exists(vit_weights_path):
            print("Downloading MedCLIP ViT pretrained weights...")
            response = requests.get(vit_weights_url)
            with open(vit_weights_path, 'wb') as f:
                f.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(vit_weights_path, 'r') as zip_ref:
            zip_ref.extractall("./medclip_vit_weights")

        # Load the weights into the model
        vit_weights_folder = "./medclip_vit_weights"
        for file in os.listdir(vit_weights_folder):
            if file.endswith(".pth"):
                weights_path = os.path.join(vit_weights_folder, file)
                self.model.vision_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                break

    def forward(self, x):
        # We are assuming x is already transformed and ready for model consumption
        pixel_values = x.to(x.device)  # Ensure the tensor is on the same device as the model

        # Extract vision features from MedCLIP
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        
        # Pass through the classifier
        logits = self.classifier(vision_outputs)
        return logits
