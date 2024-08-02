# deid-toolkit
An attempt to develop a toolkit for running and evaluating privacy preserving techniques in facial biometrics

## Toolkit Overview

- Command-line interface (something like https://docs.python.org/3.8/library/cmd.html):
  - easy to use (simple commands + names to run experiments)
  - easy to generate results
  - good look & feel
  - responsive logging (showing % of performed actions)
  - helpful tips (where results are saved, how can be visualised, etc)
- Handling of multiple separated virtual environments
  - running the models in subshells, while reporting progress to main interface
- Configuration parameters in text format via .ini file (loading & saving of different configurations within command line)
- Checking and handling of datasets
  - Raw datasets path (simple names)
  - Cropped & Aligned path (dataset preparation/standardization)
- Saving intermediate results for each phase (for each dataset, for each of the models)
- Saving final results (final deidentified images and dataset evaluation scores / plots)
- Handling and storing models / binary files for existing techniques
  - Pretrained models directory

## Toolkit Components

- Preprocess
  - Detect & Align (save transformation)
  - Image normalization
  - Video to frame conversion (for video-based datasets)
  - Annotation preparation (dataset specific scripts)
    - Identity labels parsing
    - Attribute labels parsing
    - Verification pairs generation
        - Impostor pairs
        - Genuine pairs
- Run
  - Technique 1 (separate virtual environment 1)
  - Technique 2 (separate virtual environment 2)
  - Technique 3 (separate virtual environment 3)
  - ...
- Evaluate
  - Verification
    - Model 1
    - Model 2
    - ...
  - Data utility
    - Gender
    - Age
    - Expression
    - Ethnicity
    - Attributes
  - Image quality metrics
    - MSE
    - FID
    - LPIPS
    - SSIM, MS-SSIM
  - Latent space analysis
- Visualise
  - Plot results (verification)
  - Prepare image grid
  - Webserver + interactive web interface to show generated results
- Postprocess
  - Reverse alignment
- Extras


## Set up 
1. Clone the project:
   ```sh
   git clone https://github.com/blazm/deid-toolkit
   ```

2. Get `techniques.zip` and `aligned.zip` (and `original.zip` if wanted) and extract them with unzip:
   ```sh
   unzip techniques.zip -d root_dir
   unzip aligned.zip -d root_dir/datasets
   unzip original.zip -d root_dir/datasets
   ```

3. Create the toolkit environment:
   ```sh
   conda env create -f toolkit.yml
   ```

## How to add a dataset
If the dataset is already aligned, put all images according to this structure:
```
deid-toolkit
├── root_dir
    ├── datasets
        ├── aligned
            ├── img1.jpg
            ├── img2.jpg
            ├── ...
```
If the dataset has not been preprocessed, put all images as follows:
```
deid-toolkit
├── root_dir
    ├── datasets
        ├── original
            ├── img
                ├── img1.jpg
                ├── img2.jpg
                ├── ...
```

## Labels 
Each dataset has a corresponding `.csv` file in `root_dir/datasets/labels` containing all the labels.
For each file, the headers are as follows:
```
Name,Path,Identity,Gender_code,Gender,Age,Race_code,Race,date_of_birth,Emotion_code,Neutral,Anger,Scream,Contempt,Disgust,Fear,Happy,Sadness,Surprise,Sun_glasses,Scarf,Eyeglasses,Beard,Hat,Angle
```
- `Gender_code`: Male = 1 and Female = -1
- `Emotion_code`: 0 = Neutral, 1 = Anger, 2 = Scream, 3 = Contempt, 4 = Disgust, 5 = Fear, 6 = Happy, 7 = Sadness, 8 = Surprise
- `Race_code` is not defined for now
- `Angle` :  90 = Rigth profile, 0 = Frontal, -90 = Left profile

If one of the labels isn't available for the images, leave an empty string.

## How to add a technique
Add a Python script `technique_name.py` in `root_dir/techniques`. This script must be callable with:
```sh
python technique_name.py dataset_path dataset_save_path
```

## ToolKit Command Examples

### General commands
- `root` - see currently set root directory
- `set root` - set root directory
- `serve` - run webserver to see the generated results
- `set serve` - set results directory for serving results
- `load config "filename.ini"`
- `save config "filename.ini"`
- `help "command"`
- `?` - list of all commands

### Listing commands for displaying implemented methods and current selection
- `datasets`
- `techniques`
- `evaluation`
- `visuals`
- `selection`

### Selection commands for dataset|technique|evaluation|all
- `select datasets`
- `select techniques`
- `select evaluation`
- `select *` 

### Running the processing (with feedback on the progress)
- `run preprocess`
- `run techniques`
- `run evaluation`
- `run *`
```
