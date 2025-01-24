import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

data = nib.load("M:\MRIS\data\AIBL\AD\\102\\102.nii").get_fdata()  # [:181, :217, :181]
data = data.squeeze()
#data = np.rot90(data)
#data = np.rot90(data)
#data = np.rot90(data)
#data = resize(data, (155, 240, 240), preserve_range=True)
#data = np.moveaxis(data, (0, 1, 2), (2, 1, 0))
#data = resize(data, (155, 240, 240), preserve_range=True)
plt.imshow(data[85,:,:],cmap=plt.cm.gray)
#plt.gcf().set_facecolor("red")
plt.axis("off")
plt.show()
print(data.shape)