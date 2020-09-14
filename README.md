# CDR Classification from MRI, Age, and Sex

This is a project to classify five levels of clinical dementia rating using MRI, age, and sex of the subject. The project uses CNN based model, developed using Tensorflow, to predict CDR. The trained model can also be used in transfer learning.

## Installation
### Option 1: by running commands
1. Create a new conda or docker environment with python 3.7.
2. Activate the environment.
3. Run the commands listed in `requirements.txt`.

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
### Option 1: Use command line
1. Download the model from [TODO AddPLACEHOLDER].
2. To predict CDR from MRI, age, and sex, run the command:
```
python predict.py MODEL_PATH IMAGE_PATH AGE SEX
```
Be sure to replace, `MODEL_PATH` with the path of the model (`.h5` file) you just downloaded, `IMAGE_PATH` with the `.nii` or `.nii.gz` (MRI) file, `AGE` with subject's age in years and `SEX` with `0` or `1`. Here, `0` indicates male while `1` indicates female. The output will be subject's CDR in the form of one of five numbers: 0, 0.5, 1, 2, or 3.

Note that loading a model requires a few seconds. However the actual time to predict is usually less than a second when a GPU is used.

### Option 2: Use python
1. Copy the encoder.py file in your project which can transform between one hot encoded CDR to normal CDR scale.
2. Copy the code in `predict.py` and paste it in your code.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)