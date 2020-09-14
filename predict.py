import pandas as pd
import re
import os
from pathlib import Path
import math
import numpy as np
import nibabel as nib
import sys
from tensorflow.keras.models import load_model
from encoder import Encoder

model_path = sys.argv[1]
img_path = sys.argv[2]
age_year = sys.argv[3]
sex = sys.argv[4]
enc = Encoder()

model = load_model(model_path)
img = nib.load(img_path).get_fdata()
img = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1)
cdr = model.predict({'img_input': img, 'age_input': age_year, 'gender_input': sex})
cdr = enc.decode_label(cdr)
print(cdr)