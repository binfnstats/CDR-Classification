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
import pickle

model_path = sys.argv[1]
img_path = sys.argv[2]
age_year = sys.argv[3]
sex = sys.argv[4]
label = sys.argv[5]
outdir = sys.argv[6]
enc = Encoder()

# Load model
model = load_model(model_path)

# Prepare grad model
last_layer = ''
for layer in model.layers[::-1]:
    if 'conv3d' in layer.name:
        last_layer = layer.name
        break
grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_layer).output, model.output])

# Load inputs
img = nib.load(img_path).get_fdata()
img = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1)
inp = {'img_input': img, 'age_input': age_year, 'gender_input': sex}

# Perform gradcam
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(inp)
    if label == -1:
        label = predictions.numpy()
    else:
        label = enc.encode_label(label)
    loss = predictions[:, np.argwhere(label == 1)[0][0]]
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]
weights = tf.reduce_mean(grads, axis=(0, 1, 2))
cam = np.zeros(output.shape[0:3], dtype=np.float32)
for index, w in enumerate(weights):
    cam += w * output[:, :, :, index]
    
capi = resize(cam.numpy(), (img.shape[0], img.shape[1], img.shape[2]))
heatmap = (capi - capi.min()) / (capi.max() - capi.min())
predicted_label = ds.decode_label((predictions.numpy() == np.max(predictions.numpy())).astype(int)[0])[0]
outdir = '{}/heatmap.p'.format(outdir)
pickle.dump( heatmap, open(outdir, "wb" ) )
print("Predicted CDR: {}, Heatmap file saved to {}\nUse pickle library to use the heatmap file: https://wiki.python.org/moin/UsingPickle".format(predicted_label, outdir))