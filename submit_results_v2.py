import copy

from tqdm import tqdm
import os
import time
from random import randint

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import seaborn as sns
import imageio
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output
from IPython.display import YouTubeVideo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

import albumentations as A
from albumentations import Compose, HorizontalFlip
from albumentations.pytorch import  ToTensorV2

import warnings
warnings.simplefilter("ignore")


class Image3dToGIF3d:
    """
    Displaying 3D images in 3d axes.
    Parameters:
        img_dim: shape of cube for resizing.
        figsize: figure size for plotting in inches.
    """

    def __init__(self,
                 img_dim: tuple = (55, 55, 55),
                 figsize: tuple = (15, 10),
                 binary: bool = False,
                 normalizing: bool = True,
                 ):
        """Initialization."""
        self.img_dim = img_dim
        print(img_dim)
        self.figsize = figsize
        self.binary = binary
        self.normalizing = normalizing

    def _explode(self, data: np.ndarray):
        """
        Takes: array and return an array twice as large in each dimension,
        with an extra space between each voxel.
        """
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]),
                            dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def _expand_coordinates(self, indices: np.ndarray):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    def _normalize(self, arr: np.ndarray):
        """Normilize image value between 0 and 1."""
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)

    def _scale_by(self, arr: np.ndarray, factor: int):
        """
        Scale 3d Image to factor.
        Parameters:
            arr: 3d image for scalling.
            factor: factor for scalling.
        """
        mean = np.mean(arr)
        return (arr - mean) * factor + mean

    def get_transformed_data(self, data: np.ndarray):
        """Data transformation: normalization, scaling, resizing."""
        if self.binary:
            resized_data = resize(data, self.img_dim, preserve_range=True)
            return np.clip(resized_data.astype(np.uint8), 0, 1).astype(np.float32)

        norm_data = np.clip(self._normalize(data) - 0.1, 0, 1) ** 0.4
        scaled_data = np.clip(self._scale_by(norm_data, 2) - 0.1, 0, 1)
        resized_data = resize(scaled_data, self.img_dim, preserve_range=True)

        return resized_data

    def plot_cube(self,
                  cube,
                  title: str = '',
                  init_angle: int = 0,
                  make_gif: bool = False,
                  path_to_save: str = 'filename.gif'
                  ):
        """
        Plot 3d data.
        Parameters:
            cube: 3d data
            title: title for figure.
            init_angle: angle for image plot (from 0-360).
            make_gif: if True create gif from every 5th frames from 3d image plot.
            path_to_save: path to save GIF file.
            """
        if self.binary:
            facecolors = cm.winter(cube)
            print("binary")
        else:
            if self.normalizing:
                cube = self._normalize(cube)
            facecolors = cm.gist_stern(cube)
            print("not binary")

        facecolors[:, :, :, -1] = cube
        facecolors = self._explode(facecolors)

        filled = facecolors[:, :, :, -1] != 0
        x, y, z = self._expand_coordinates(np.indices(np.array(filled.shape) + 1))

        with plt.style.context("dark_background"):

            fig = plt.figure(figsize=self.figsize)
            ax = fig.gca(projection='3d')

            ax.view_init(30, init_angle)
            ax.set_xlim(right=self.img_dim[0] * 2)
            ax.set_ylim(top=self.img_dim[1] * 2)
            ax.set_zlim(top=self.img_dim[2] * 2)
            ax.set_title(title, fontsize=18, y=1.05)

            ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

            if make_gif:
                images = []
                for angle in tqdm(range(0, 360, 5)):
                    ax.view_init(30, angle)
                    fname = str(angle) + '.png'

                    plt.savefig(fname, dpi=120, format='png', bbox_inches='tight')
                    images.append(imageio.imread(fname))
                    # os.remove(fname)
                imageio.mimsave(path_to_save, images)
                plt.close()

            else:
                plt.show()


class ShowResult:

    def mask_preprocessing(self, mask):
        """
        Test.
        """
        mask = mask.squeeze().cpu().detach().numpy()
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        mask_WT = np.rot90(montage(mask[0]))
        mask_TC = np.rot90(montage(mask[1]))
        mask_ET = np.rot90(montage(mask[2]))

        return mask_WT, mask_TC, mask_ET

    def image_preprocessing(self, image):
        """
        Returns image flair as mask for overlaping gt and predictions.
        """
        image = image.squeeze().cpu().detach().numpy()
        image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))
        flair_img = np.rot90(montage(image[0]))
        return flair_img

    def plot(self, image, ground_truth, prediction):
        image = self.image_preprocessing(image)
        gt_mask_WT, gt_mask_TC, gt_mask_ET = self.mask_preprocessing(ground_truth)
        pr_mask_WT, pr_mask_TC, pr_mask_ET = self.mask_preprocessing(prediction)

        fig, axes = plt.subplots(1, 2, figsize=(35, 30))

        [ax.axis("off") for ax in axes]
        axes[0].set_title("Ground Truth", fontsize=35, weight='bold')
        axes[0].imshow(image, cmap='bone')
        axes[0].imshow(np.ma.masked_where(gt_mask_WT == False, gt_mask_WT),
                       cmap='cool_r', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_TC == False, gt_mask_TC),
                       cmap='autumn_r', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_ET == False, gt_mask_ET),
                       cmap='autumn', alpha=0.6)

        axes[1].set_title("Prediction", fontsize=35, weight='bold')
        axes[1].imshow(image, cmap='bone')
        axes[1].imshow(np.ma.masked_where(pr_mask_WT == False, pr_mask_WT),
                       cmap='cool_r', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_TC == False, pr_mask_TC),
                       cmap='autumn_r', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_ET == False, pr_mask_ET),
                       cmap='autumn', alpha=0.6)

        plt.tight_layout()

        plt.show()


# show_result = ShowResult()
# show_result.plot(data['image'], data['mask'], data['mask'])


def merging_two_gif(path1: str, path2: str, name_to_save: str):
    """
    Merging GIFs side by side.
    Parameters:
        path1: path to gif with ground truth.
        path2: path to gif with prediction.
        name_to_save: name for saving new GIF.
    """
    # https://stackoverflow.com/questions/51517685/combine-several-gif-horizontally-python
    # Create reader object for the gif
    gif1 = imageio.get_reader(path1)
    gif2 = imageio.get_reader(path2)

    # If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length())

    # Create writer object
    new_gif = imageio.get_writer(name_to_save)

    for frame_number in range(number_of_frames):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        # here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()


# merging_two_gif('BraTS20_Training_001_flair_3d.gif',
#                'BraTS20_Training_001_flair_3d.gif',
#                'result.gif')

def get_all_csv_file(root: str) -> list:
    """Extraction all unique ids from file names."""
    ids = []
    for dirname, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            if path.endswith(".csv"):
                ids.append(path)
    ids = list(set(filter(None, ids)))
    print(f"Extracted {len(ids)} csv files.")
    return ids


# csv_paths = get_all_csv_file("../input/brats20-dataset-training-validation/BraTS2020_TrainingData")

# %%

class GlobalConfig:
    root_dir = '/home/zengsn/BRATS/Data/Brats2020/'
    train_root_dir = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    path_to_csv = 'test_data.csv'
    pretrained_model_path = '/home/zengsn/BRATS/Data/Brats2020/brats2020logs/brats2020logs/unet/last_epoch_model.pth'
    train_logs_path = '/home/zengsn/BRATS/Data/Brats2020//brats2020logs/brats2020logs/unet/train_log.csv'
    ae_pretrained_model_path = '/home/zengsn/BRATS/Data/Brats2020/brats2020logs/brats2020logs/ae/autoencoder_best_model.pth'
    tab_data = '/home/zengsn/BRATS/Data/Brats2020/brats2020logs/brats2020logs/data/df_with_voxel_stats_and_latent_features.csv'
    seed = 55


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# %%


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
            img_2 = self.load_img("/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz")
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


# %%

def get_augmentations(phase):
    list_transforms = []

    list_trfms = Compose(list_transforms)
    return list_trfms




def sample_test():
    sample_filename = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz'
    sample_filename_mask = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz'

    sample_img = nib.load(sample_filename)
    sample_img = np.asanyarray(sample_img.dataobj)
    sample_img = np.rot90(sample_img)
    sample_mask = nib.load(sample_filename_mask)
    sample_mask = np.asanyarray(sample_mask.dataobj)
    sample_mask = np.rot90(sample_mask)
    print("img shape ->", sample_img.shape)
    print("mask shape ->", sample_mask.shape)

    # %%

    sample_filename2 = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz'
    sample_img2 = nib.load(sample_filename2)
    sample_img2 = np.asanyarray(sample_img2.dataobj)
    sample_img2 = np.rot90(sample_img2)

    sample_filename3 = '/home/zengsn/BRATS/Data/Brats2020//BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz'
    sample_img3 = nib.load(sample_filename3)
    sample_img3 = np.asanyarray(sample_img3.dataobj)
    sample_img3 = np.rot90(sample_img3)

    sample_filename4 = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz'
    sample_img4 = nib.load(sample_filename4)
    sample_img4 = np.asanyarray(sample_img4.dataobj)
    sample_img4 = np.rot90(sample_img4)

    mask_WT = sample_mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1

    mask_TC = sample_mask.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 4] = 1

    mask_ET = sample_mask.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 4] = 1

    # %% md

    ### What's the data looks like ?

    # %%
    # https://matplotlib.org/3.3.2/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
    # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
    fig = plt.figure(figsize=(20, 10))

    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    flair = ax0.imshow(sample_img[:, :, 65], cmap='bone')
    ax0.set_title("FLAIR", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(flair)

    #  Varying density along a streamline
    ax1 = fig.add_subplot(gs[0, 1])
    t1 = ax1.imshow(sample_img2[:, :, 65], cmap='bone')
    ax1.set_title("T1", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t1)

    #  Varying density along a streamline
    ax2 = fig.add_subplot(gs[0, 2])
    t2 = ax2.imshow(sample_img3[:, :, 65], cmap='bone')
    ax2.set_title("T2", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t2)

    #  Varying density along a streamline
    ax3 = fig.add_subplot(gs[0, 3])
    t1ce = ax3.imshow(sample_img4[:, :, 65], cmap='bone')
    ax3.set_title("T1 contrast", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t1ce)

    #  Varying density along a streamline
    ax4 = fig.add_subplot(gs[1, 1:3])

    # ax4.imshow(np.ma.masked_where(mask_WT[:,:,65]== False,  mask_WT[:,:,65]), cmap='summer', alpha=0.6)
    l1 = ax4.imshow(mask_WT[:, :, 65], cmap='summer', )
    l2 = ax4.imshow(np.ma.masked_where(mask_TC[:, :, 65] == False, mask_TC[:, :, 65]), cmap='rainbow', alpha=0.6)
    l3 = ax4.imshow(np.ma.masked_where(mask_ET[:, :, 65] == False, mask_ET[:, :, 65]), cmap='winter', alpha=0.6)

    ax4.set_title("", fontsize=20, weight='bold', y=-0.1)

    _ = [ax.set_axis_off() for ax in [ax0, ax1, ax2, ax3, ax4]]

    colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3]]
    labels = ['Non-Enhancing tumor core', 'Peritumoral Edema ', 'GD-enhancing tumor']
    patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, fontsize='xx-large',
               title='Mask Labels', title_fontsize=18, edgecolor="black", facecolor='#c5c6c7')

    plt.suptitle("Multimodal Scans -  Data | Manually-segmented mask - Target", fontsize=20, weight='bold')

    fig.savefig("data_sample.png", format="png", pad_inches=0.2, transparent=False, bbox_inches='tight')
    fig.savefig("data_sample.svg", format="svg", pad_inches=0.2, transparent=False, bbox_inches='tight')

    # %%

    YouTubeVideo('nrmizEvG8aM', width=600, height=400)

def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(probabilities: torch.Tensor,
                        truth: torch.Tensor,
                        treshold: float = 0.5,
                        eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


class Meter:
    '''factory for storing and updating iou and dice scores.'''

    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)

    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        return dice, iou


class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert (probability.shape == targets.shape)

        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        # print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert (logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)

        return bce_loss + dice_loss


# helper functions for testing.
def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                 truth: np.ndarray,
                                 treshold: float = 0.5,
                                 eps: float = 1e-9,
                                 classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Dice score for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with dice scores for each class.
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


def jaccard_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Jaccard index for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with jaccard scores for each class."
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


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

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask


def compute_scores_per_classes(model,
                               dataloader,
                               classes):
    """
    Compute Dice and Jaccard coefficients for each class.
    Params:
        model: neural net for make predictions.
        dataloader: dataset object to load data from.
        classes: list with classes.
        Returns: dictionaries with dice and jaccard coefficients for each class for each slice.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    print(dice_scores_per_classes)
    exit(1)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            dice_scores = dice_coef_metric_per_classes(logits, targets)
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)

            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes


def compute_results(model,
                    dataloader,
                    treshold=0.33):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {"Id": [], "image": [], "GT": [], "Prediction": []}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            id_, imgs, targets = data['Id'], data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            predictions = (probs >= treshold).float()
            predictions = predictions.cpu()
            targets = targets.cpu()

            results["Id"].append(id_)
            results["image"].append(imgs.cpu())
            results["GT"].append(targets)
            results["Prediction"].append(predictions)

            # only 5 pars
            if (i > 5):
                return results
        return results


def get_augmentations(phase):
    list_transforms = []

    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
        dataset: torch.utils.data.Dataset,
        path_to_csv: str,
        phase: str,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 4
):
    '''Returns: dataloader for the model training'''


    df = pd.read_csv(path_to_csv)

    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader


# %%

class AutoEncoderDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "test"):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)

            img = self.normalize(img)
            images.append(img.astype(np.float32))
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        return {
            "Id": id_,
            "data": img,
            "label": img,
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray, mean=0.0, std=1.0):
        """Normilize image value between 0 and 1."""
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)



#Trainer AutoEncoder
class Trainer2:
    def __init__(self,
                 net: nn.Module,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 fold: int,
                 num_epochs: int,
                 path_to_csv: str,
                 dataset: torch.utils.data.Dataset,
                 ):

        """Initialization."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=2, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs

        self.dataloaders = {

            phase: get_dataloader(
                dataset=dataset,
                path_to_csv=path_to_csv,
                phase=phase,
                fold=fold,
                batch_size=batch_size,
                num_workers=4
            )
            for phase in self.phases
        }
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch['data'], data_batch['label']
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches

        self.losses[phase].append(epoch_loss)
        print(f"Loss | {self.losses[phase][-1]}")

        return epoch_loss

    def run(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                print(f"\n{'#' * 20}\nSaved new checkpoint\n{'#' * 20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "autoencoder_best_model.pth")
            print()
        self._save_train_history()

    def load_predtrain_model(self,
                             state_path: str):
        self.net.load_state_dict(torch.load(state_path, map_location=device))
        print("Predtrain model loaded")

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(self.net.state_dict(),
                   f"autoencoder_last_epoch_model.pth")

# %%
class LatentFeaturesGenerator:
    def __init__(self,
                 autoencoder,
                 device: str = 'cuda'):
        self.autoencoder = autoencoder.to(device)
        self.device = device

    def __call__(self, img):
        with torch.no_grad():
            img = torch.FloatTensor(img).unsqueeze(0).to(self.device)
            latent_features = self.autoencoder.encode(
                img, return_partials=False).squeeze(0).cpu().numpy()

        return latent_features


# %%

class Features_Generator:

    def __init__(self, df, autoencoder):
        self.df = df
        self.df_voxel_stats = pd.DataFrame()
        self.latent_feature_generator = LatentFeaturesGenerator(autoencoder)

    def _read_file(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj).astype(np.float32)
        return data

    def _normalize(self, data: np.ndarray):
        """Normilize image value between 0 and 1."""
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def _create_features(self, Brats20ID):
        features = {}
        images = []
        # vOXEL STATS
        for data_type in ['_t1.nii.gz', '_t2.nii.gz', '_flair.nii.gz', '_t1ce.nii.gz']:

            # data path
            root_path = self.df.loc[self.df['Brats20ID'] == Brats20ID]['path'].values[0]
            file_path = os.path.join(root_path, Brats20ID + data_type)

            # flatten 3d array
            img_data = self._read_file(file_path)
            data = img_data.reshape(-1)

            # create features
            data_mean = data.mean()
            data_std = data.std()
            intensive_data = data[data > data_mean]
            more_intensive_data = data[data > data_mean + data_std]
            non_intensive_data = data[data < data_mean]

            data_skew = stats.skew(data)
            data_kurtosis = stats.kurtosis(data)
            intensive_skew = stats.skew(intensive_data)
            non_intensive_skew = stats.skew(non_intensive_data)

            data_diff = np.diff(data)

            # write new features in df
            features['Brats20ID'] = Brats20ID
            features[f'{data_type}_skew'] = data_skew,
            features[f'{data_type}_kurtosis'] = data_kurtosis,
            features[f'{data_type}_diff_skew'] = stats.skew(data_diff),
            features[f'{data_type}_intensive_dist'] = intensive_data.shape[0],
            features[f'{data_type}_intensive_skew'] = intensive_skew,
            features[f'{data_type}_non_intensive_dist'] = non_intensive_data.shape[0],
            features[f'{data_type}_non_intensive_skew'] = non_intensive_skew,
            # features[f'{data_type}_intensive_non_intensive_mean_ratio'] = intensive_data.mean() / non_intensive_data.mean(),
            # features[f'{data_type}_intensive_non_intensive_std_ratio'] = intensive_data.std() / non_intensive_data.std(),
            features[f'{data_type}_data_intensive_skew_difference'] = data_skew - intensive_skew,
            features[f'{data_type}_data_non_intensive_skew_difference'] = data_skew - non_intensive_skew,
            features[f'{data_type}_more_intensive_dist'] = more_intensive_data.shape[0],

            parts = 15
            for p, part in enumerate(np.array_split(data, parts)):
                features[f'{data_type}_part{p}_mean'] = part.mean()

            # Latent Features
            img = self._normalize(img_data)
            images.append(img.astype(np.float32))

        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        latent_features = self.latent_feature_generator(img)

        for i, lf in enumerate(latent_features):
            features[f'latent_f{i}'] = lf

        return pd.DataFrame(features)

    def run(self):

        for _, row in tqdm(self.df.iterrows()):
            ID = row['Brats20ID']

            df_features = self._create_features(ID)

            self.df_voxel_stats = pd.concat([self.df_voxel_stats, df_features], axis=0)

        self.df_voxel_stats.reset_index(inplace=True, drop=True)
        self.df_voxel_stats = self.df_voxel_stats.merge(self.df[['Brats20ID', 'Age', 'Survival_days']], on='Brats20ID',
                                                        how='left')

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))


def compute_val_metrics(net, data):

    meter = Meter()
    vdataloader = data
    total_batches = len(vdataloader)
    running_loss = 0.0
    net = net.to(device).double()
    criterion = BCEDiceLoss()

    with torch.no_grad():
        for itr, data_batch in enumerate(vdataloader):
            images = data_batch['image']
            images = images.to(torch.device("cpu")).double()
            logits = net(images)
            print(logits)
            #loss = criterion(logits, targets)
            # running_loss += loss.item()
            # meter.update(logits.detach().cpu(),
            #              targets.detach().cpu()
            #              )

    val_loss = (running_loss) / total_batches
    val_dice, val_iou = meter.get_metrics()


    return val_loss,val_dice,val_iou



def dice_test(probabilities: np.ndarray,
                                 truth: np.ndarray,
                                 ) -> np.ndarray:
            eps: float = 1e-9
            prediction = probabilities
            truth_ = truth
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores = 1.0
            else:
                scores = (intersection + eps) / union

            return scores


def save_labels(model, dataloader, save_path):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_et = []

    data = nib.load(
        "/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_003/BraTS20_Training_003_seg.nii.gz")
    affine_f = data.affine


    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs = data['image'].float()

            id_path = save_path+str(data['Id'][0])+".nii.gz"
            imgs = imgs.to(device)
            logits = model(imgs)
            logits = torch.sigmoid(logits)
            logits = logits.detach().cpu().numpy()
            print("old shape", logits.shape)
            logits = np.moveaxis(logits, (0, 1, 2, 3, 4), (0, 1, 4, 3, 2))
            logits = np.squeeze(logits)

            wt = logits[0, :, :, :]
            tc = logits[1, :, :, :]
            et = logits[2, :, :, :]

            full_image = np.zeros_like(wt, dtype=np.uint8)
            et_indexes = np.where(et >= 0.80)
            tc_indexes = np.where(tc >= 0.80)
            wt_indexes = np.where(wt >= 0.80)

            #full_image[et_indexes] = 4

            df_et = pd.DataFrame({"x": et_indexes[0], "y": et_indexes[1], "z": et_indexes[2]}).reset_index()
            df_tc = pd.DataFrame({"x": tc_indexes[0], "y": tc_indexes[1], "z": tc_indexes[2]}).reset_index()

            result = pd.merge(df_et, df_tc, how="outer", on=["x", "y", "z"], indicator=True)
            tc_vals = result[result['_merge'] == 'right_only']
            both_et = result[result['_merge'] == 'both']
            both_et = both_et[["x", "y", "z"]].reset_index()


            #tc_vals_index = (tc_vals['x'].to_numpy(), tc_vals['y'].to_numpy(), tc_vals['z'].to_numpy())
            #full_image[tc_vals_index] = 1


            df_wt = pd.DataFrame({"x": wt_indexes[0], "y": wt_indexes[1], "z": wt_indexes[2]}).reset_index()
            main_indexes = np.where((full_image == 4) & (full_image == 1))
            df_main = pd.DataFrame({"x": main_indexes[0], "y": main_indexes[1], "z": main_indexes[2]}).reset_index()

            result_main = pd.merge(df_main, df_wt, how="outer", on=["x", "y", "z"], indicator=True)
            merge_both_et_all = pd.merge(both_et, df_wt, how="outer", on=["x", "y", "z"], indicator=True)
            both_et_final = merge_both_et_all[merge_both_et_all['_merge'] == 'both']
            et_all_both_index = (both_et_final['x'].to_numpy(), both_et_final['y'].to_numpy(), both_et_final['z'].to_numpy())

            # get tc
            result_tc = pd.merge(df_tc, df_wt, how="outer", on=["x", "y", "z"], indicator=True)
            both_tc = result_tc[result_tc['_merge'] == 'both']
            both_tc = both_tc[["x", "y", "z"]].reset_index()

            result_all_tc = pd.merge(both_tc, df_et, how="outer", on=["x", "y", "z"], indicator=True)
            result_all_tc_vals = result_all_tc[result_all_tc['_merge'] == 'left_only']
            result_all_tc_indexes = (result_all_tc_vals['x'].to_numpy(), result_all_tc_vals['y'].to_numpy(), result_all_tc_vals['z'].to_numpy())
            full_image[result_all_tc_indexes] = 1


            wt_vals = result_main[result_main['_merge'] == 'right_only']

            wt_vals_index = (wt_vals['x'].to_numpy(), wt_vals['y'].to_numpy(), wt_vals['z'].to_numpy())
            full_image[wt_vals_index] = 2

            # wt_dice = dice_test(full_image, target_wt)
            # total_et.append(wt_dice)
            # print("tc dice ", np.mean(total_et))
            # continue

            full_image[et_all_both_index] = 4
            print("My image ", full_image.shape)
            new_image = nib.Nifti1Image(full_image, affine_f)
            print("My image ", new_image.shape)
            nib.save(new_image, id_path)








def evaluate_trained_model():

    np.random.seed(45)
    val_dataloader = get_dataloader(BratsDataset, 'test_data.csv', phase='test', fold=0)

    # data = nib.load("/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_003/BraTS20_Training_003_seg.nii.gz")
    # #data = nib.load("/home/zengsn/BRATS/Data/SegmentationLabels/FedProx/BraTS20_Validation_001.nii.gz")
    # affine_f = data.affine
    # data = np.asarray(data.dataobj)
    # print(type(affine_f))
    # exit(1)

    global_model = UNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
    global_model.load_state_dict(torch.load("raw_FedProx/last_FedProx_round_model.pth", map_location=device))
    global_model.eval()

    save_labels(global_model, val_dataloader, "/home/zengsn/BRATS/Data/SegmentationLabels/FedProx2/")




if __name__ == "__main__":

    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    config = GlobalConfig()
    seed_everything(config.seed)

    name_mapping_df = pd.read_csv(
        '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/name_mapping_validation_data.csv')

    name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True)

    df = name_mapping_df

    paths = []
    for _, row in df.iterrows():

        id_ = row['Brats20ID']
        phase = id_.split("_")[-2]
        path = os.path.join(config.test_root_dir, id_)
        paths.append(path)

    df['path'] = paths
    test_data = df

    print("test_df ->", test_data.shape)
    test_data.to_csv("test_data.csv", index=False)

    #Dataset DataLoader
    evaluate_trained_model()






