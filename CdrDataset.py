import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import math
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from encoder import Encoder

class CdrDataset:

    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df = self.df[self.df['age_year'].notnull()]
        self.df['age_year'] = self.df['age_year'].astype(int)
        self.df['age_year'] = (self.df['age_year']/10).astype(int)*10
        self.df['dataset_subject'] = self.df['dataset'] + "_" + self.df['subject']
        self.df.loc[self.df['subject'] == 'OAS30948', 'month_diff'] = 30
        self.df = self.df[self.df['month_diff'] <= 36]
        self.encoder = Encoder()
        
    def get_full_dataset(self):
        return self.df

    def get_train_test(self, datasets, include_augmented, test_size):
        df2 = self.df.copy()
        if len(datasets) > 0:
            df2 = df2[df2['dataset'].isin(datasets)]
        df3 = df2[df2['augmented'] == False]

        # Train test split.
        df_train, df_test = train_test_split(df3, test_size=test_size, random_state=42, stratify=df3['cdr'])
        
        # Add augmented samples to the dataset.
        if include_augmented == True:
            subs = df_train['dataset_subject'].unique()
            df_aug = df2[df2['augmented'] == True]
            for dataset_subject in subs:
                df_aug_sub = df_aug[df_aug['dataset_subject'] == dataset_subject]
                df_train = pd.concat([df_train, df_aug_sub], axis=0)
            df_train = df_train.sample(frac=1)
        return df_train, df_test

    def encode_label(self, cdr, is_binary):
        return self.encoder.encode_label(cdr, is_binary)

    def decode_label(self, cdr):
        return self.encoder.decode_label(cdr)
    
    def read_mri_pickle(self, path):
        return pickle.load(open(path, 'rb'))

    def get_sample(self, row, is_binary):
        
        # Read image
        img = self.read_mri_pickle(row['mri_path'])
        
        # Encode label, age, gender
        label = self.encoder.encode_label(row['cdr'], is_binary)
        gender = 1
        if row['sex'] == 'male':
            gender = 0
        gender = np.array(gender).reshape((1,))
        age = np.array(row['age_year']).reshape((1,))
        return {'img_input': img, 'age_input': age, 'gender_input': gender}, label
    
    def plot_mri(self, mri, slice_x=50, slice_y=50, slice_z=50, title=''):
        if len(mri.shape) == 4:
            mri = mri[:, :, :, 0]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13,4))
        slice_x, slice_y, slice_z = int(mri.shape[0] * slice_x / 100), int(mri.shape[1] * slice_y / 100), int(mri.shape[2] * slice_z / 100)
        ax1.imshow(mri[:, :, slice_z])
        ax1.set_axis_off()
        ax2.imshow(mri[:, slice_y, :])
        ax2.set_axis_off()
        ax3.imshow(mri[slice_x, :, :])
        ax3.set_axis_off()
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()