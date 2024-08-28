import torch.nn as nn
import torch
import numpy as np
import math

# simplified ViT (using average value for class token)
class SimpViT(nn.Module):
    def __init__(self, 
                 img_size=64, 
                 patch_size=16, 
                 num_outputs=2, 
                 hidden_size=64, 
                 num_layers=2, 
                 num_heads=4
                ):
        super(SimpViT, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(in_channels=2, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_outputs)
        
    def forward(self, x):
        patches = self.patch_embedding(x)  # [batch_size, hidden_size, num_patches, num_patches]
        patches = patches.permute(0, 2, 3, 1)  # [batch_size, num_patches, num_patches, hidden_size]
        patches = patches.view(x.size(0), -1, patches.size(-1))  # [batch_size, num_patches*num_patches, hidden_size]
        
        patches = patches + self.position_embedding[:, :patches.size(1)]
        
        patches = patches.permute(1, 0, 2)  # [num_patches*num_patches, batch_size, hidden_size]
        encoded_patches = self.transformer_encoder(patches)
        
        class_token = encoded_patches.mean(dim=0)  # [batch_size, hidden_size]
        
        output = self.fc(class_token)
        return output
    
# transform the conversion data into image tensor (2*img_size*img_size)
def transform(data, img_size=64):
    img_arr = np.zeros((2, img_size, img_size))
    for f1, total_conv, conv1, conv2 in data:
        # Ensure that the index does not go out of bounds
        row = min(math.floor(f1 * img_size), img_size - 1)
        col = min(math.floor(total_conv * img_size), img_size - 1)

        img_arr[0][row][col] = conv1
        img_arr[1][row][col] = conv2
    return torch.tensor(img_arr).float()

# simplified ViT (using average value for class token)
class SimpViT_3D(nn.Module):
    def __init__(self, 
                 img_size=64, 
                 patch_size=16, 
                 num_outputs=6, 
                 hidden_size=64, 
                 num_layers=2, 
                 num_heads=4
                ):
        super(SimpViT_3D, self).__init__()
        self.num_patches = (img_size // patch_size) ** 3
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv3d(in_channels=3, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_outputs)
        
    def forward(self, x):
        patches = self.patch_embedding(x)  # [batch_size, hidden_size, num_patches, num_patches, num_patches]
        patches = patches.permute(0, 2, 3, 4, 1)  # [batch_size, num_patches, num_patches, num_patches, hidden_size]
        patches = patches.view(x.size(0), -1, patches.size(-1))  # [batch_size, num_patches^3, hidden_size]
        
        patches = patches + self.position_embedding[:, :patches.size(1)]
        
        patches = patches.permute(1, 0, 2)  # [num_patches*^3, batch_size, hidden_size]
        encoded_patches = self.transformer_encoder(patches)
        
        class_token = encoded_patches.mean(dim=0)  # [batch_size, hidden_size]
        
        output = self.fc(class_token)
        return output

def transform_ternary(data, img_size=64):
    # Check if data is a list and convert to NumPy array if necessary
    if isinstance(data, list):
        data = np.array(data)
    
    # Check the shape of the data
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError("Input data must be a two-dimensional array with exactly 6 columns.")

    # Initialize the image array
    img_arr = np.zeros((3, img_size, img_size, img_size), dtype=np.float32)
    
    # Calculate indices ensuring they are within bounds
    X = np.clip((data[:, 0] * img_size).astype(int), 0, img_size - 1)
    Y = np.clip((data[:, 1] * img_size).astype(int), 0, img_size - 1)
    Z = np.clip((data[:, 2] * img_size).astype(int), 0, img_size - 1)
    
    # Fill image array using advanced indexing
    idx = (X, Y, Z)
    img_arr[0][idx] = data[:, 3]
    img_arr[1][idx] = data[:, 4]
    img_arr[2][idx] = data[:, 5]

    # Convert to a PyTorch tensor
    return torch.tensor(img_arr, dtype=torch.float32)
