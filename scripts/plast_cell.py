import numpy as np
import PIL
from PIL import Image
import pims
from glob import glob
data_dir = "data"
ext = ".lif"
glob_str = f"{data_dir}/**/*{ext}"
files = glob(glob_str,recursive=True)

# https://github.com/soft-matter/pims/pull/403
pims.bioformats.download_jar(version="6.7.0")
                        
# ims = [pims.Bioformats(file) for file in files]

print("ok")
im = pims.Bioformats(files[0])