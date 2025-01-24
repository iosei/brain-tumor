import copy

from tqdm import tqdm
import os
import time
from random import randint

import numpy as np
from scipy import stats
import pandas as pd
import multiprocessing
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
import io
import os
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

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import imageio
    from tqdm import tqdm

    def plot_cube(self,
                  cube,
                  title: str = '',
                  init_angle: int = 0,
                  make_gif: bool = False,
                  path_to_save: str = 'filename.gif'):
        """
        Plot 3D data.

        Parameters:
            cube: 3D data.
            title: Title for the figure.
            init_angle: Initial angle for the 3D plot (from 0-360 degrees).
            make_gif: If True, create a GIF by rotating the plot and capturing every 5th frame.
            path_to_save: Path to save the GIF file.
        """
        if self.binary:
            facecolors = cm.winter(cube)
            print("binary")
        else:
            if self.normalizing:
                cube = self._normalize(cube)
            facecolors = cm.gist_stern(cube)
            print("not binary")

        # Ensure the alpha channel is correctly assigned
        facecolors[:, :, :, -1] = cube

        # Explode the facecolors (ensure this method preserves the intended shape)
        facecolors = self._explode(facecolors)

        # Create filled mask for the voxels (True where cube is not zero)
        filled = facecolors[:, :, :, -1] != 0

        # Create voxel coordinates (ensure shapes match)
        # np.indices creates a grid of indices with shape (3, 240, 240, 156)
        # self._expand_coordinates should process this into x, y, z with shape (240, 240, 156)
        indices = np.indices(np.array(filled.shape) + 1)
        x, y, z = self._expand_coordinates(indices)

        # Corrected Sanity Check
        expected_shape = tuple(s + 1 for s in filled.shape)
        if not (x.shape == y.shape == z.shape == expected_shape):
            raise ValueError(
                f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}, "
                f"expected_shape={expected_shape}, filled.shape={filled.shape}"
            )

        with plt.style.context("dark_background"):
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')

            # Set initial view angle
            ax.view_init(30, init_angle)
            ax.set_xlim(right=self.img_dim[0] * 2)
            ax.set_ylim(top=self.img_dim[1] * 2)
            ax.set_zlim(top=self.img_dim[2] * 2)
            ax.set_title(title, fontsize=18, y=1.05)

            # Plot the voxels
            ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

            if make_gif:
                images = []
                for angle in tqdm(range(0, 360, 5)):
                    ax.view_init(30, angle)
                    fname = f"{angle}.png"

                    plt.savefig(fname, dpi=120, format='png', bbox_inches='tight')
                    images.append(imageio.imread(fname))
                    # Optionally remove temp files
                    # os.remove(fname)
                imageio.mimsave(path_to_save, images)
                plt.close()
            else:
                plt.show()

    # %%
    import os
    import imageio
    import numpy as np


def merging_two_gif(path1: str, path2: str, name_to_save: str):
        """
        Merges two GIFs side by side.

        Parameters:
            path1 (str): Path to the first GIF (e.g., ground truth).
            path2 (str): Path to the second GIF (e.g., prediction).
            name_to_save (str): Name for saving the new merged GIF.
        """
        # Print and validate input paths
        print(f"Checking if the file exists at path1: {os.path.abspath(path1)}")
        print(f"Checking if the file exists at path2: {os.path.abspath(path2)}")

        if not os.path.isfile(path1):
            raise FileNotFoundError(f"The file at path1 does not exist: {path1}")
        if not os.path.isfile(path2):
            raise FileNotFoundError(f"The file at path2 does not exist: {path2}")

        try:
            # Create reader objects for the GIFs
            gif1 = imageio.get_reader(path1)
            gif2 = imageio.get_reader(path2)

            # Check the number of frames in each GIF
            number_of_frames1 = gif1.get_length()
            number_of_frames2 = gif2.get_length()

            # Take the shorter of the two GIFs
            number_of_frames = min(number_of_frames1, number_of_frames2)

            if number_of_frames == 0:
                raise ValueError("One or both of the GIFs have no frames.")

            # Create a writer object for the new GIF
            with imageio.get_writer(name_to_save) as new_gif:
                for frame_number in range(number_of_frames):
                    try:
                        img1 = gif1.get_data(frame_number)
                        img2 = gif2.get_data(frame_number)

                        # Check if both frames have the same height
                        if img1.shape[0] != img2.shape[0]:
                            raise ValueError("The GIFs have different heights and cannot be merged side by side.")

                        # Merge the frames side by side
                        new_image = np.hstack((img1, img2))
                        new_gif.append_data(new_image)

                    except IndexError as e:
                        print(f"Frame {frame_number} could not be read: {e}")
                        break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            gif1.close()
            gif2.close()


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
    root_dir = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//'
    train_root_dir = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_TrainingData//MICCAI_BraTS2020_TrainingData'
    test_root_dir = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_ValidationData//MICCAI_BraTS2020_ValidationData'
    path_to_csv = 'test_data.csv'
    pretrained_model_path = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//brats2020logs//last_epoch_model.pth'
    train_logs_path = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//brats2020logs//train_log.csv'
    ae_pretrained_model_path = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//brats2020logs//autoencoder_best_model.pth'
    tab_data = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//brats2020logs//df_with_voxel_stats_and_latent_features.csv'
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

    list_trfms = Compose(list_transforms,is_check_shapes=False)
    return list_trfms





def sample_test():
    sample_filename = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_TrainingData//MICCAI_BraTS2020_TrainingData//BraTS20_Training_001//BraTS20_Training_001_flair.nii.gz'
    sample_filename_mask = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_TrainingData//MICCAI_BraTS2020_TrainingData//BraTS20_Training_001//BraTS20_Training_001_seg.nii.gz'

    sample_img = nib.load(sample_filename)
    sample_img = np.asanyarray(sample_img.dataobj)
    sample_img = np.rot90(sample_img)
    sample_mask = nib.load(sample_filename_mask)
    sample_mask = np.asanyarray(sample_mask.dataobj)
    sample_mask = np.rot90(sample_mask)
    print("img shape ->", sample_img.shape)
    print("mask shape ->", sample_mask.shape)

    # %%

    sample_filename2 = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_TrainingData//MICCAI_BraTS2020_TrainingData//BraTS20_Training_001//BraTS20_Training_001_t1.nii.gz'
    sample_img2 = nib.load(sample_filename2)
    sample_img2 = np.asanyarray(sample_img2.dataobj)
    sample_img2 = np.rot90(sample_img2)

    sample_filename3 = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_TrainingData//MICCAI_BraTS2020_TrainingData//BraTS20_Training_001//BraTS20_Training_001_t2.nii.gz'
    sample_img3 = nib.load(sample_filename3)
    sample_img3 = np.asanyarray(sample_img3.dataobj)
    sample_img3 = np.rot90(sample_img3)

    sample_filename4 = 'C://Users//NEW//Desktop//FedTumor//FedTumor//BRATS//Data//Brats2020//BraTS2020_TrainingData//MICCAI_BraTS2020_TrainingData//BraTS20_Training_001//BraTS20_Training_001_t1ce.nii.gz'
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

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import DetrModel, DetrConfig
#
#
# class DetrBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.adjust_channels = nn.Conv2d(in_channels, 3, kernel_size=1)
#
#         # Initialize DETR model
#         config = DetrConfig()
#         self.detr = DetrModel(config)
#
#         # Adjust the output channels
#         self.conv1x1 = nn.Conv3d(9, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         b, c, d, h, w = x.shape
#         x = x.permute(0, 2, 1, 3, 4)  # Change to (batch, depth, channels, height, width)
#         x = x.reshape(b * d, c, h, w)  # Combine batch and depth dimensions
#         # print("original shape ",x.size())
#         # Apply the 2D DETR to each slice
#         # Resize each slice to the expected input size of the DETR
#         x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
#         x = self.adjust_channels(x)
#
#         # Flatten the image for DETR input
#         x = x.view(b * d, 3, -1).permute(0, 2, 1)  # (batch, sequence_length, channels)
#         # print("size of input", x.unsqueeze(0).size())
#         x = self.detr(pixel_values=x.unsqueeze(0).permute(0,3,1,2)).last_hidden_state
#         x = x.view(1,1,1,100,256)
#         # Reshape back to original dimensions
#         #x = x.permute(0, 2, 1).view(b * d, config.hidden_dim, 28, 28)  # (batch * depth, channels, height, width)
#         x = F.interpolate(x, size=(d, h, w*192), mode='trilinear', align_corners=False)
#         # print("shapes after all",x.size())
#         x = x.view(1,9, 192, 15, 15)
#         #x = x.permute(0, 2, 1, 3, 4)  # Back to (batch, channels, depth, height, width)
#         #x = x.permute(0, 2, 1, 3, 4)  # Back to (batch, channels, depth, height, width)
#
#         # Adjust channels if necessary
#         x = self.conv1x1(x)
#
#         return x
#

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
        self.detr_block=DetrBlock(8*n_channels,8*n_channels)


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
        # Apply DETR block
        # print("mu initial size",x5.size())
        detr=self.detr_block(x5)



        mask = self.dec1(detr, x4)
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
    device = 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    print(dice_scores_per_classes)


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
    device = 'cpu'
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



def get_dataloader(
        dataset: torch.utils.data.Dataset,
        path_to_csv: str,
        phase: str,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 4,
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
        self.device = 'cpu'
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

def evaluate_trained_model():

    np.random.seed(45)
    val_dataloader = get_dataloader(dataset=BratsDataset, path_to_csv='train_data.csv', phase='valid', fold=0)
    no_sample_data = len(val_dataloader)
    print(no_sample_data)

    data = next(iter(val_dataloader))
    print(data['Id'], data['image'].shape)


    img_tensor = data['image'].squeeze()[0].cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())


    image = np.rot90(montage(img_tensor))


    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image, cmap='bone')
    plt.show()

    global_model = UNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)

    global_model.load_state_dict(torch.load("C://Users//NEW//Desktop//135//last_FedperAvg_round_model.pth", map_location=device), strict=False)


    #global val metrics
    global_model.eval()
    #val_loss, val_dice, val_iou = compute_val_metrics(global_model, val_dataloader)

    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        global_model, val_dataloader, ['WT', 'TC', 'ET']
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
    ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15);
    ax.set_title("Dice and Jaccard Coefficients from Validation", fontsize=20)

    for idx, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(100 * val_metics_df.mean().values[idx])
        x = p.get_x() + p.get_width() / 2 - 0.15
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    fig.savefig("singleno_result_per.png", format="png", pad_inches=0.2, transparent=False, bbox_inches='tight')
    fig.savefig("singleno_result_per.svg", format="svg", pad_inches=0.2, transparent=False, bbox_inches='tight')

    results = compute_results(
        global_model, val_dataloader, 0.33)

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

    title = "SinglemGround Truth_fed" + id_[0]
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

    title = "Single Prediction_fed" + id_[0]
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
                    'result.gif')
    show_gif('Single result.gif', format='png')



if __name__ == "__main__":


    #sample_test()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    gpu = 0
    device = torch.device('cpu')
    config = GlobalConfig()
    seed_everything(config.seed)

    #Dataset DataLoader
    evaluate_trained_model()






