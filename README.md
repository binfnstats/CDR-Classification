# CDR Classification from MRI, Age, and Sex

This is a project to classify five levels of clinical dementia rating using MRI, age, and sex of the subject. The project uses CNN based model, developed using Tensorflow, to predict CDR. The trained model can also be used in transfer learning.

## Installation
### Option 1: by running commands
1. Create a new conda or docker environment with python 3.7.
2. Activate the environment.
3. Run the commands listed in [`requirements.txt`](https://github.com/binfnstats/CDR-Classification/blob/master/requirements.txt).
4. Install [niftyreg](https://github.com/KCL-BMEIS/niftyreg/wiki/install).
5. Download the [standard MNI space MRI](http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09c_nifti.zip).

### Option 2: by building identical conda environment
Run the command
```
conda create --name myenv --file spec-file.txt
```

## Getting data
Download the required data from the folder `/project/RDS-SMS-FFgenomics-RW/raquib/cdr/` of Artemis to the local `data` folder.

## Training your own model
To start the training, run the command
```
python train.py LOGDIR
```
Be sure to replace `LOGDIR` with the path of your log directory where tensorboard history and model will be saved.

## Use our trained model to predict CDR
1. Register your mri image into standard MNI space [ICBM 2009c Nonlinear Symmetric MNI space](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009).
    ```
    reg_aladin -flo PATH_TO_YOUR_MRI -ref PATH_TO_MNI_FILE -res OUTPUT_MRI_PATH
    ```
    Be sure to replace `PATH_TO_YOUR_MRI` with the path to your MRI file, `PATH_TO_MNI_FILE` path to the MNI file you downloaded, and `OUTPUT_MRI_PATH` location where the registered MRI should be saved.
2. To classify the MRI, follow one of the following two options.

### Option 1: Use command line
1. Download the model from [TODO AddPLACEHOLDER].
2. To predict CDR from MRI, age, and sex, run the command:
```
python predict.py MODEL_PATH IMAGE_PATH AGE SEX
```
Be sure to replace, `MODEL_PATH` with the path of the model (`.h5` file) you just downloaded, `IMAGE_PATH` with the `.nii` or `.nii.gz` (MRI) file, `AGE` with subject's age in years and `SEX` with `0` or `1`. Here, `0` indicates male while `1` indicates female. The output will be subject's CDR in the form of one of five numbers: 0, 0.5, 1, 2, or 3.

Note that loading a model requires a few seconds. However the actual time to predict is usually less than a second when a GPU is used.

### Option 2: Use python
1. Copy the [`encoder.py`](https://github.com/binfnstats/CDR-Classification/blob/master/encoder.py) file in your project which can transform between one hot encoded CDR to normal CDR scale.
2. Copy the code in [`predict.py`](https://github.com/binfnstats/CDR-Classification/blob/master/predict.py) and paste it in your code.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)