import torch.nn as nn 
from torchvision.models import resnet18, resnet50, vit_b_16, ViT_B_16_Weights
from transformers import ViTModel, ViTFeatureExtractor
import torch
from timm import create_model
import logging
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel  
from medclip import MedCLIPProcessor
import requests
import zipfile
import os

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
