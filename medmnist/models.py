import torch.nn as nn 
from torchvision.models import resnet18, resnet50, vit_b_16, ViT_B_16_Weights
from transformers import ViTModel, ViTFeatureExtractor
import torch
from timm import create_model
# from einops import rearrange
import logging
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel  
from medclip import MedCLIPProcessor
import requests
import zipfile
import os
# from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter  # Import conversion classes
from torch.nn import BatchNorm2d as SynchronizedBatchNorm2d
from torch.nn import BatchNorm3d as SynchronizedBatchNorm3d


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
# def convert_to_acs_or_conv3d(model, conv_type='ACSConv'):
#     if conv_type == 'ACSConv':
#         model = ACSConverter(model)
#     elif conv_type == 'Conv2_5d':
#         model = Conv2_5dConverter(model)
#     elif conv_type == 'Conv3d':
#         model = Conv3dConverter(model)
#     return model

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(VisionTransformer, self).__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

class VisionTransformerTimm(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(VisionTransformerTimm, self).__init__()
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes) #added this head for fine tuning 

    def forward(self, x):
        return self.vit(x)

#vit timm for 3D
# class ViT3D(nn.Module):
#     def __init__(self, num_classes, pretrained, patch_size=16, embedding_dim=768):
#         super(ViT3D, self).__init__()
#         self.patch_size = patch_size
#         self.embedding_dim = embedding_dim
#         self.num_patches = (28 // patch_size) ** 3  # Assuming the input is 28x28x28
#         self.projection = nn.Linear(patch_size ** 3, embedding_dim)
#         self.vit = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)

#     def forward(self, x):
#         # Convert 3D volume into patches
#         patches = self._generate_patches(x)

#         # Project the patches into the embedding space
#         embeddings = self.projection(patches)

#         # Feed into the ViT model
#         return self.vit(embeddings)

#     def _generate_patches(self, x):
#         # Create patches from the 3D volume
#         B, C, D, H, W = x.shape
#         x = rearrange(x, 'b c (d p1) (h p2) (w p3) -> b (d h w) (c p1 p2 p3)', p1=self.patch_size, p2=self.patch_size, p3=self.patch_size)
#         return x

class ViT3D(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(ViT3D, self).__init__()
        self.vit = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)  # Set num_classes to 0 to remove default head
        self.head = nn.Linear(self.vit.num_features, num_classes)  # Define your custom head

    def forward(self, x):
        x = self.vit(x)
        x = self.head(x)  # Apply the custom head
        return x


class VisionTransformerHuggingFace(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(VisionTransformerHuggingFace, self).__init__()
        if pretrained:
            logger.info("Loading pretrained Hugging Face ViT model")
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            logger.info("Loading non-pretrained Hugging Face ViT model")
            self.vit = ViTModel(config=ViTModel.config_class())
        
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x).last_hidden_state
        logits = self.classifier(outputs[:, 0, :])
        return logits

class MedCLIPViTModel(nn.Module):
    def __init__(self, num_classes):
        super(MedCLIPViTModel, self).__init__()
        # Initialize the model
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

        # Load the pretrained weights manually
        self.load_pretrained_vit_weights()

        # Add a final classification layer based on MedCLIP's architecture
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

# class MedCLIPViTModel(nn.Module):
#     def __init__(self, num_classes):
#         super(MedCLIPViTModel, self).__init__()
#         # Initialize the processor and model
#         self.processor = MedCLIPProcessor()
#         self.model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
#         self.model.from_pretrained()  # Load the pretrained MedCLIP model

#         # Add a final classification layer
#         self.classifier = nn.Linear(self.model.vision_model.config.hidden_size, num_classes)

#     def forward(self, x):
#         # Process the input and move it to the appropriate device
#         with torch.no_grad():
#             inputs = self.processor(images=x, return_tensors="pt")
        
#         # Extract vision features from MedCLIP
#         pixel_values = inputs['pixel_values'].to(x.device)  # Ensure the tensor is on the same device as the model
#         vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        
#         # Pass through the classifier
#         logits = self.classifier(vision_outputs.pooler_output)
#         return logits
