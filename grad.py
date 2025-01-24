import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import Resize
from imageio import imread
from scipy.ndimage import zoom
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the DoubleConv, Down, Up, Out, and UNet3D classes

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DetrModel, DetrConfig


class DetrBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adjust_channels = nn.Conv2d(in_channels, 3, kernel_size=1)

        # Initialize DETR model
        config = DetrConfig()
        self.detr = DetrModel(config)

        # Adjust the output channels
        self.conv1x1 = nn.Conv3d(9, out_channels, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # Change to (batch, depth, channels, height, width)
        x = x.reshape(b * d, c, h, w)  # Combine batch and depth dimensions
        # print("original shape ",x.size())
        # Apply the 2D DETR to each slice
        # Resize each slice to the expected input size of the DETR
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.adjust_channels(x)

        # Flatten the image for DETR input
        x = x.view(b * d, 3, -1).permute(0, 2, 1)  # (batch, sequence_length, channels)
        # print("size of input", x.unsqueeze(0).size())
        x = self.detr(pixel_values=x.unsqueeze(0).permute(0,3,1,2)).last_hidden_state
        x = x.view(1,1,1,100,256)
        # Reshape back to original dimensions
        #x = x.permute(0, 2, 1).view(b * d, config.hidden_dim, 28, 28)  # (batch * depth, channels, height, width)
        x = F.interpolate(x, size=(d, h, w*192), mode='trilinear', align_corners=False)
        # print("shapes after all",x.size())
        x = x.view(1,9, 192, 15, 15)
        #x = x.permute(0, 2, 1, 3, 4)  # Back to (batch, channels, depth, height, width)
        #x = x.permute(0, 2, 1, 3, 4)  # Back to (batch, channels, depth, height, width)

        # Adjust channels if necessary
        x = self.conv1x1(x)

        return x


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        # DETR block
        self.detr_block = DetrBlock(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        detr = self.detr_block(x5)

        mask = self.dec1(detr, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask




# Define the dataset class for BraTS 2020

class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "test", is_resize: bool = False):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
        self.is_resize = is_resize

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        #print("Hmmm ", root_path)
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)  # .transpose(2, 0, 1)
            img_2 = self.load_img("C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_TrainingData//MICCAI_BraTS2020_TrainingData//BraTS20_Training_001//BraTS20_Training_001_seg.nii.gz")
            #img_r = img_2[img_2==2]
            #print(img_r)

            if self.is_resize:
                img = self.resize(img)

            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        print("My id ", id_)
        if self.phase != "test":
            mask_path = os.path.join(root_path, id_ + "_seg.nii.gz")
            mask = self.load_img(mask_path)

            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)

            augmented = self.augmentations(image=img.astype(np.float32),
                                           mask=mask.astype(np.float32))

            img = augmented['image']
            mask = augmented['mask']

            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }

        return {
            "Id": id_,
            "image": img,

        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


    def get_augmentations(self, phase):
        # Define any augmentations for training phase here
        if phase == "train":
            return None  # Replace with actual augmentations
        return None


# Initialize the UNet3D model
in_channels = 4  # 4 input modalities: FLAIR, T1, T1ce, T2
n_classes = 3    # 3 output classes
n_channels = 24  # Number of base channels
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu = 0
device = torch.device('cpu')
# Load the model weights
model = UNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
model.load_state_dict(torch.load("last_FedDyn_round_50_model.pth", map_location=device))
model.eval()

# Grad-CAM Visualization
def run_gradcam_on_image(image_path):
    # Load and preprocess the input image
    image = nib.load(image_path).get_fdata()
    image = image.astype(np.float32)

    device = torch.device('cpu')

    # Normalize and resize the image
    image = (image - np.mean(image)) / np.std(image)
    image = resize(image, (1, 78, 120, 120), preserve_range=True)  # Assuming input shape for the model

    input_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    # Initialize Grad-CAM
    target_layers = [model.enc4]

    device = torch.device('cpu')
    # Adjust the target layer as needed
    cam = GradCAM(model=model, target_layers=target_layers)

    # Generate Grad-CAM
    targets = [ClassifierOutputTarget(2)]  # Change the target class as needed
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Assuming single image

    # Visualization
    visualization = show_cam_on_image(image[0, ...], grayscale_cam, use_rgb=False)  # Adjust for RGB if needed
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig('grad_cam_visualization.png', bbox_inches='tight')
    plt.show()


# Example usage
image_path = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_TrainingData//MICCAI_BraTS2020_TrainingData//BraTS20_Training_001//BraTS20_Training_001_seg.nii.gz'  # Change to your image path
run_gradcam_on_image(image_path)
