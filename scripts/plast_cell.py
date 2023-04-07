import numpy as np
import PIL
from PIL import Image
import pims
from glob import glob
data_dir = "data"
ext = ".lif"
glob_str = f"{data_dir}/**/*{ext}"
files = glob(glob_str,recursive=True)
ims = [pims.Bioformats(file) for file in files]

print("ok")
