#using this one version for pretrained 
# import torch.nn as nn
# from einops import rearrange, repeat
# import torch
# import numpy as np

# class Embedding(nn.Module):
#     def __init__(
#         self,
#         image_size=28,
#         patch_size=7,  # Adjust patch size as needed
#         num_patches=8,  # Adjust based on patch size
#         channels=3,  # Number of input channels (3 for RGB)
#         embedding_dim=768,  # Adjust based on your model complexity
#     ):
#         super(Embedding, self).__init__()

#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         # Linear layer to project patches to the embedding dimension
#         self.projectionModel = nn.Linear(patch_size**3 * channels, embedding_dim)

#         # Positional embedding (1, num_patches + 1, embedding_dim) (+1 for the class token)
#         self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

#         # Class token embedding
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

#     def _generate_patches(self, x):
#         # Expecting input shape: (B, C, D, H, W)
#         B, C, D, H, W = x.shape
#         patch_d = patch_h = patch_w = self.patch_size

#         # Break the image into patches and flatten them
#         patches = x.unfold(2, patch_d, patch_d).unfold(3, patch_h, patch_h).unfold(4, patch_w, patch_w)
#         patches = patches.contiguous().view(B, C, -1, patch_d * patch_h * patch_w)
#         patches = patches.permute(0, 2, 1, 3)  # Reorder to (B, num_patches, C, flattened_patch_size)
#         patches = patches.contiguous().view(B, -1, C * patch_d * patch_h * patch_w)  # Flatten each patch

#         return patches

#     def forward(self, x):
#         x = self._generate_patches(x)  # Generate patches and flatten them
#         x = self.projectionModel(x)  # Project patches to the embedding dimension
#         b, _, _ = x.shape
#         cls_token = repeat(self.cls_token, " () s e -> b s e", b=b)
#         x = torch.cat([cls_token, x], dim=1)  # Add class token at the beginning
#         x = x + self.pos_embed  # Add positional embeddings
#         return x

#this emb works fine but not if using pretrained 
import torch.nn as nn
from einops import rearrange, repeat
import torch
import numpy as np

class Embedding(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=7,  # Adjust patch size as needed
        num_patches=8,  # Adjust based on patch size
        channels=3,  # Number of input channels (3 for RGB)
        embedding_dim=256,  # Adjust based on your model complexity
        linear_patch=False,
        position_embedding_dropout=None,
        cls_head=True,
        verbose=1,
    ):
        super(Embedding, self).__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.verbose = verbose
        self.channels = channels

        def _positional_embedding(seq_len, emb_size):
            embeddings = torch.ones(seq_len, emb_size)
            for i in range(seq_len):
                for j in range(emb_size):
                    embeddings[i][j] = (
                        np.sin(i / (pow(10000, j / emb_size)))
                        if j % 2 == 0
                        else np.cos(i / (pow(10000, (j - 1) / emb_size)))
                    )
            return embeddings

        self.projectionModel = nn.Linear(patch_size**3 * channels, embedding_dim)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_dim))

        self.pos_embed = nn.Parameter(
            _positional_embedding(self.num_patches + 1, embedding_dim)
        )

    def _generate_patches(self, x):
        # Expecting input shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        patch_d = patch_h = patch_w = self.patch_size

        patches = []
        for i in range(0, D, patch_d):
            for j in range(0, H, patch_h):
                for k in range(0, W, patch_w):
                    patch = x[:, :, i : i + patch_d, j : j + patch_h, k : k + patch_w]
                    patches.append(patch)

        patches = torch.stack(patches)  # Convert list of patches to a single tensor
        patches = rearrange(patches, "p b c d h w -> b p (c d h w)")

        return patches

    def forward(self, x):
        x = self._generate_patches(x)
        x = self.projectionModel(x)
        b, _, _ = x.shape
        cls_token = repeat(self.cls_token, " () s e -> b s e", b=b)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        return x