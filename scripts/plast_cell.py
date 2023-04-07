import numpy as np
import PIL
from PIL import Image
import pims
from glob import glob
import matplotlib.pyplot as plt
data_dir = "data"
import bioimage_phenotyping as bip
from bip.segmentation import Segmenter

# Lif files are the brightfield images 
ext = ".lif"
glob_str = f"{data_dir}/**/*{ext}"
files = glob(glob_str,recursive=True)

# https://github.com/soft-matter/pims/pull/403
pims.bioformats.download_jar(version="6.7.0")
                        
# ims = [pims.Bioformats(file) for file in files]

print("ok")
im = pims.Bioformats(files[0])

# DAPI

im.default_coords['c'] = 0



# Brightfield

im.default_coords['c'] = 1

plt.imshow(im[0])

image_2D = im[0]



Segmenter()
image_2D

    

