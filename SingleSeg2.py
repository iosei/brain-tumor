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
from albumentations.pytorch import ToTensorV2

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
    root_dir = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020'
    train_root_dir = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    path_to_csv = 'train_data.csv'
    pretrained_model_path = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/brats2020logs/brats2020logs/unet/last_epoch_model.pth'
    train_logs_path = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/brats2020logs/brats2020logs/unet/train_log.csv'
    ae_pretrained_model_path = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/brats2020logs/brats2020logs/ae/autoencoder_best_model.pth'
    tab_data = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/brats2020logs/brats2020logs/data/df_with_voxel_stats_and_latent_features.csv'
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
        # print("Hmmm ", root_path)
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)  # .transpose(2, 0, 1)

            if self.is_resize:
                img = self.resize(img)

            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.phase != "test":
            mask_path = os.path.join(root_path, id_ + "_seg.nii.gz")
            mask = self.load_img(mask_path)

            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)
            # print("image shape",img.shape,mask.shape)
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

    list_trfms = Compose(list_transforms, is_check_shapes=False)
    return list_trfms


def get_dataloader(
        dataset: torch.utils.data.Dataset,
        path_to_csv: str,
        phase: str,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 4,
        is_check_shapes=False,
):
    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)

    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if phase == "train" else val_df
    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


def sample_test():
    sample_filename = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz'
    sample_filename_mask = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz'

    sample_img = nib.load(sample_filename)
    sample_img = np.asanyarray(sample_img.dataobj)
    sample_img = np.rot90(sample_img)
    sample_mask = nib.load(sample_filename_mask)
    sample_mask = np.asanyarray(sample_mask.dataobj)
    sample_mask = np.rot90(sample_mask)
    print("img shape ->", sample_img.shape)
    print("mask shape ->", sample_mask.shape)

    # %%

    sample_filename2 = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz'
    sample_img2 = nib.load(sample_filename2)
    sample_img2 = np.asanyarray(sample_img2.dataobj)
    sample_img2 = np.rot90(sample_img2)

    sample_filename3 = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz'
    sample_img3 = nib.load(sample_filename3)
    sample_img3 = np.asanyarray(sample_img3.dataobj)
    sample_img3 = np.rot90(sample_img3)

    sample_filename4 = 'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz'
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


#########################
###########################

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.transformer import TransformerEncoderLayer


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, channels, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                TransformerEncoderLayer(
                    d_model=channels * dim_head,
                    nhead=heads,
                    dim_feedforward=mlp_dim
                ),
                nn.Conv3d(channels, channels, kernel_size=1)
            ))

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = rearrange(x, 'b c d h w -> (b d) (h w) c')
        for layer in self.layers:
            x = layer(x)
        x = rearrange(x, '(b d) (h w) c -> b c d h w', b=b, h=h, w=w)
        return x


class TransformerUNet3d(nn.Module):
    def __init__(self, in_channels, out_channels, features, transformer_depth=1, heads=8, dim_head=64, mlp_dim=2048):
        super().__init__()
        self.down1 = ConvBlock(in_channels, features)
        self.down2 = ConvBlock(features, features * 2)
        self.down3 = ConvBlock(features * 2, features * 4)

        self.transformer = TransformerBlock(features * 4, transformer_depth, heads, dim_head, mlp_dim)

        self.up1 = ConvBlock(features * 4, features * 2)
        self.up2 = ConvBlock(features * 2, features)

        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool3d(d1, kernel_size=2, stride=2))
        d3 = self.down3(F.max_pool3d(d2, kernel_size=2, stride=2))

        t = self.transformer(d3)

        u1 = self.up1(F.interpolate(t, scale_factor=2, mode='trilinear', align_corners=True))
        u2 = self.up2(F.interpolate(u1, scale_factor=2, mode='trilinear', align_corners=True))

        return self.final_conv(u2)


#############################
#############################


class Trainer:
    """
    Factory for training proccess.
    Args:
        display_plot: if True - plot train history after each epoch.
        net: neural network for mask prediction.
        criterion: factory for calculating objective loss.
        optimizer: optimizer for weights updating.
        phases: list with train and validation phases.
        dataloaders: dict with data loaders for train and val phases.
        path_to_csv: path to csv file.
        meter: factory for storing and updating metrics.
        batch_size: data batch size for one step weights updating.
        num_epochs: num weights updation for all data.
        accumulation_steps: the number of steps after which the optimization step can be taken
                    (https://www.kaggle.com/c/understanding_cloud_organization/discussion/105614).
        lr: learning rate for optimizer.
        scheduler: scheduler for control learning rate.
        losses: dict for storing lists with losses for each phase.
        jaccard_scores: dict for storing lists with jaccard scores for each phase.
        dice_scores: dict for storing lists with dice scores for each phase.
    """

    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 fold: int,
                 num_epochs: int,
                 path_to_csv: str,
                 display_plot: bool = True,
                 ):

        """Initialization."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
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
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}

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
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch['image'], data_batch['mask']
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(),
                         targets.detach().cpu()
                         )

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)

        return epoch_loss

    def run(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)
            if self.display_plot:
                pass
                # self._plot_train_history()

            if val_loss < self.best_loss:
                print(f"\n{'#' * 20}\nSaved new checkpoint\n{'#' * 20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "best_model.pth")
            print()
        self._save_train_history()

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,

            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]} 
            """,

            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['val'][-1]}
            """,
        ]

        clear_output(True)
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

    def load_predtrain_model(self,
                             state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(self.net.state_dict(),
                   f"last_epoch_model.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                for key in logs_[i]]
        log_names = [key + log_names_[i]
                     for i in list(range(len(logs_)))
                     for key in logs_[i]
                     ]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv("raw_train_log_single.csv", index=False)


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


def train_val_3unet(dataloader):
    data = next(iter(dataloader))
    data['Id'], data['image'].shape, data['mask'].shape
    print("shape", data['image'].shape, data['mask'].shape)
    img_tensor = data['image'].squeeze()[0].cpu().detach().numpy()
    mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))

    image = np.rot90(montage(img_tensor))
    mask = np.rot90(montage(mask_tensor))

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image, cmap='bone')
    ax.imshow(np.ma.masked_where(mask == False, mask),
              cmap='cool', alpha=0.6)
    # plt.show()

    # nodel = UNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
    nodel = TransformerUNet3d(in_channels=1, out_channels=1, features=64)
    # %%
    trainer = Trainer(net=nodel,
                      dataset=BratsDataset,
                      criterion=BCEDiceLoss(),
                      lr=5e-4,
                      accumulation_steps=4,
                      batch_size=1,
                      fold=0,
                      num_epochs=400,
                      path_to_csv=config.path_to_csv, )

    if config.pretrained_model_path is not None:
        # trainer.load_predtrain_model(config.pretrained_model_path)
        pass
    # if config.pretrained_model_path is not None:
    #     trainer.load_predtrain_model(config.pretrained_model_path, device=device)
    #
    #     # if need - load the logs.
    #     train_logs = pd.read_csv(config.train_logs_path)
    #     trainer.losses["train"] = train_logs.loc[:, "train_loss"].to_list()
    #     trainer.losses["val"] = train_logs.loc[:, "val_loss"].to_list()
    #     trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
    #     trainer.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
    #     trainer.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
    #     trainer.jaccard_scores["val"] = train_logs.loc[:, "val_jaccard"].to_list()

    trainer.run()

    # End of UNet

    val_dataloader = get_dataloader(BratsDataset, 'train_data.csv', phase='valid', fold=0)
    len(dataloader)

    nodel.eval()
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        nodel, val_dataloader, ['WT', 'TC', 'ET']
    )

    # %%

    dice_df = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

    iou_df = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']
    val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df = val_metics_df.loc[:, ['WT dice', 'WT jaccard',
                                          'TC dice', 'TC jaccard',
                                          'ET dice', 'ET jaccard']]
    val_metics_df.sample(5)

    # %%

    colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    palette = sns.color_palette(colors, 6)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=val_metics_df.mean().index, y=val_metics_df.mean(), palette=palette, ax=ax)
    ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15)
    ax.set_title("Dice and Jaccard Coefficients from Validation", fontsize=20)

    for idx, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
        x = p.get_x() + p.get_width() / 2 - 0.15
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    fig.savefig("single_result1.png", format="png", pad_inches=0.2, transparent=False, bbox_inches='tight')
    fig.savefig("single_result1.svg", format="svg", pad_inches=0.2, transparent=False, bbox_inches='tight')

    results = compute_results(
        nodel, val_dataloader, 0.33)

    # %%
    for id_, img, gt, prediction in zip(results['Id'][4:],
                                        results['image'][4:],
                                        results['GT'][4:],
                                        results['Prediction'][4:]
                                        ):
        print(id_)
        break

    ### convert 3d to 2d ground truth and prediction
    show_result = ShowResult()
    show_result.plot(img, gt, prediction)

    ### 3d binary mask projection for ground truth and prediction
    gt = gt.squeeze().cpu().detach().numpy()
    gt = np.moveaxis(gt, (0, 1, 2, 3), (0, 3, 2, 1))
    wt, tc, et = gt
    print(wt.shape, tc.shape, et.shape)
    gt = (wt + tc + et)
    gt = np.clip(gt, 0, 1)
    print(gt.shape)

    # %%

    title = "Ground Truth_single_" + id_[0]
    filename1 = title + "_3d.gif"

    data_to_3dgif = Image3dToGIF3d(img_dim=(120, 120, 78), binary=True, normalizing=False)
    transformed_data = data_to_3dgif.get_transformed_data(gt)
    data_to_3dgif.plot_cube(
        transformed_data,
        title=title,
        make_gif=True,
        path_to_save=filename1
    )

    # show_gif(filename1, format='png'
    prediction = prediction.squeeze().cpu().detach().numpy()
    prediction = np.moveaxis(prediction, (0, 1, 2, 3), (0, 3, 2, 1))
    wt, tc, et = prediction
    print(wt.shape, tc.shape, et.shape)
    prediction = (wt + tc + et)
    prediction = np.clip(prediction, 0, 1)
    print(prediction.shape)

    # %%

    title = "Prediction_single_" + id_[0]
    filename2 = title + "_3d.gif"

    data_to_3dgif = Image3dToGIF3d(img_dim=(120, 120, 78), binary=True, normalizing=False)
    transformed_data = data_to_3dgif.get_transformed_data(prediction)
    data_to_3dgif.plot_cube(
        transformed_data,
        title=title,
        make_gif=True,
        path_to_save=filename2
    )

    # show_gif(filename2, format='png')#
    merging_two_gif(filename1,
                    filename2,
                    'result_single.gif')
    show_gif('result_single.gif', format='png')


if __name__ == "__main__":

    sample_test()
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    config = GlobalConfig()
    seed_everything(config.seed)

    survival_info_df = pd.read_csv(
        'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv')
    name_mapping_df = pd.read_csv(
        'C:/Users/HP/PycharmProjects/FedTumor/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv')

    name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True)

    df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")

    paths = []
    for _, row in df.iterrows():

        id_ = row['Brats20ID']
        phase = id_.split("_")[-2]

        if phase == 'Training':
            path = os.path.join(config.train_root_dir, id_)
        else:
            path = os.path.join(config.test_root_dir, id_)
        paths.append(path)

    df['path'] = paths

    # split data on train, test, split
    # train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=69, shuffle=True)
    # train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
    train_data["Age_rank"] = train_data["Age"] // 10 * 10
    train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, )

    skf = StratifiedKFold(
        n_splits=7, random_state=config.seed, shuffle=True
    )
    for i, (train_index, val_index) in enumerate(
            skf.split(train_data, train_data["Age_rank"])
    ):
        train_data.loc[val_index, "fold"] = i

    train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
    val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)

    test_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
    print("train_df ->", train_df.shape, "val_df ->", val_df.shape, "test_df ->", test_df.shape)
    train_data.to_csv("train_data.csv", index=False)

    # Dataset DataLoader
    dataloader = get_dataloader(dataset=BratsDataset, path_to_csv='train_data.csv', phase='valid', fold=0,
                                is_check_shapes=False)
    print("data length", len(dataloader))
    train_val_3unet(dataloader)


