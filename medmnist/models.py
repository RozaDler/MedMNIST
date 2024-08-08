import torch.nn as nn 
from torchvision.models import resnet18, resnet50, vit_b_16, ViT_B_16_Weights
from transformers import ViTModel, ViTFeatureExtractor
from timm import create_model
import logging

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