import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

class Encoder:

    def __init__(self):
        labels = np.array([0, 0.5, 1, 2, 3]).astype(str)
        self.label_enc = LabelEncoder()
        int_encoded = self.label_enc.fit_transform(labels)
        int_encoded = int_encoded.reshape(len(int_encoded), 1)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(int_encoded)
        
    def encode_label(self, cdr, is_binary):
        if is_binary:
            if cdr > 0:
                return 1
            return 0
        else:
            encoded = self.label_enc.transform([str(float(cdr))])
            encoded = encoded.reshape(len(encoded), 1)
            return self.enc.transform(encoded).toarray()[0]

    def decode_label(self, cdr):
        decoded = self.enc.inverse_transform([cdr])
        return self.label_enc.inverse_transform(decoded[0])