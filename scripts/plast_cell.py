# from torchvision.datasets import ImageFolder
# from torchvision.transforms import Compose, Normalize
import urllib.request
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pims
import torch
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

data_dir = "data"
import io

import bioimageio.core
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from albumentations import Compose
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, data, exposure, io
from skimage.color import label2rgb
from skimage.morphology import disk
from skimage.segmentation import mark_boundaries

import bioimage_phenotyping as bip
from bioimage_phenotyping.segmentation import WatershedSegmenter

# Overlay the segmentation results on the original image

# Lif files are the brightfield images
ext = ".lif"
glob_str = f"{data_dir}/**/*{ext}"
files = glob(glob_str, recursive=True)

# https://github.com/soft-matter/pims/pull/403
pims.bioformats.download_jar(version="6.7.0")

# ims = [pims.Bioformats(file) for file in files]

print("ok")
im = pims.Bioformats(files[0])

# DAPI

# im.default_coords['c'] = 0

# # Brightfield

# im.default_coords['c'] = 1
im.default_coords["c"] = 0
image_2D = im[0]
plt.imshow(image_2D)

out = WatershedSegmenter()(image_2D)
plt.imshow(out)
plt.show()

print(out)

# # Brightfield

im.default_coords["c"] = 1
image_2D = Image.fromarray(np.array(im[0]))
plt.imshow(image_2D)


# LiveCell pretrained Unet from bioimage.io
rdf = "https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.5869899/6647688/rdf.yaml"
model_resource = bioimageio.core.load_resource_description(rdf)


model = torch.jit.load(model_resource.weights["torchscript"].source)


# TODO Unsure of how to automate this from the model resource description
transform = A.Compose(
    [
        A.ToGray(),
        A.Resize(512, 512, always_apply=True),
        # A.Lambda(lambda x: np.expand_dims(x, axis=0)),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ]
)

model.eval()
inputs = np.array(image_2D)

# np.array(image_2D)
transformed = transform(image=inputs)["image"]

# https://docs.monai.io/en/stable/inferers.html
with torch.no_grad():
    outputs = model(transformed.unsqueeze(0))

# %% Plotting
image = exposure.equalize_hist(inputs)
labeled_coins, _ = ndi.label(outputs[0][1] > 0.333)
image_label_overlay = label2rgb(labeled_coins, image=image, bg_label=0)

# Display the results
plt.imshow(image_label_overlay)
plt.show()
