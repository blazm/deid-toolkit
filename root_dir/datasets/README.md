# Datasets
This module manages the datasets required for de-identification. It’s the first part of the pipeline. The toolkit is able to integrate (<-----) additional facial images datasets. Moreover, the datasets won’t be included in the toolkit because some of them have different licensing constraints.

## Table of contents
1. Basic usage 
2. How to integrate a dataset

## Basic usage
* `select datasets`: Run this command to start the selection
<!-- TODO: add a image to show how the selection should be (with the numbers)-->

## How to add a dataset

### 1 Insert the images in the toolkit
To add a dataset, the toolkit accepts it in one of the following three cases:
* **Already aligned dataset**
If the dataset is put all images according to this structure:

```plaintext
📂 deid-toolkit
├── 📁 root_dir               # Main project directory
│   ├── 📂 datasets           
│   │   ├── 📂 aligned        
│   │   │   ├── 📂 datasetName  # Dataset-specific folder
│   │   │   │   ├── 🖼️ img1.jpg   # Example aligned image 1
│   │   │   │   ├── 🖼️ img2.jpg   # Example aligned image 2
│   │   │   │   ├── ...

```
* **Not preprocessed dataset**
If the dataset has not been preprocessed, put all images as follows:
```plaintext
📂 deid-toolkit
├── 📁 root_dir                     # Main project directory
│   ├── 📁 datasets                 # Datasets directory
│   │   ├── 📁 original             # Original, unprocessed dataset
│   │   │   ├── 📂 datasetName  # Dataset-specific folder
│   │   │   │   ├── 📁 img              # Image files for analysis
│   │   │   │   │   ├── 🖼️ img1.jpg     # Example image 1
│   │   │   │   │   ├── 🖼️ img2.jpg     # Example image 2
│   │   │   │   │   ├── ...             
```
* **Aligned and deidentified dataset** 
If you have the aligned dataset and deidentified images you can add them accouding to this structure:
```plaintext
📂 deid-toolkit
├── 📁 root_dir               # Main project directory
│   ├── 📂 datasets           
│   │   ├── 📂 aligned        # Aligned images for processing
│   │   │   ├── 📂 datasetName  # Dataset-specific folder
│   │   │   │   ├── 🖼️ img1.jpg   
│   │   │   │   ├── 🖼️ img2.jpg   
│   │   │   │   ├── ...
│   │   ├── 📂 technique      # Deidentified images directory
│   │   │   ├── 🖼️ img1.jpg   # Example deidentified image 1
│   │   │   ├── 🖼️ img2.jpg   # Example deidentified image 2
│   │   │   ├── ...

```
> [!IMPORTANT]
> The technique directory **must be named as same as** the technique.py filename in `./root_dir/techniques/technique_name`
> For example:
> ```plaintext
> 📂 deid-toolkit
> ├── 📁 root_dir               
> │   ├── 📂 datasets           
> │   │   ├── 📂 aligned        
> │   │   │   ├── 📂 datasetName  # Dataset-specific folder for aligned images
> │   │   │   │   ├── 🖼️ img1.jpg   
> │   │   │   │   ├── 🖼️ img2.jpg   
> │   │   │   │   ├── ...
> │   │   ├── 📂 blur              # Directory for deidentified images (blur technique)
> │   │   │   ├── 📂 datasetName    # Dataset-specific folder for blur images
> │   │   │   │   ├── 🖼️ deidimg1.jpg   
> │   │   │   │   ├── 🖼️ deidimg2.jpg   
> │   │   ├── 📂 techniques           
> │   │   │   ├── blur.py           # Script that defines the blur technique
> 
> ```
> The new technique file must exist, even if empty, to allow the toolkit to select it. For more details go to *how to add a technique*
>
<!-- TODO: add link in the word "how to add add technique" to learn how-->


> [!TIP]
> To save disk space is not neccesary keep the original copy of the original dataset. We recommend to `zip` the original dataset after preprocessing it. 

### 2 Labels

The labels enable the toolkit to perform the correct evaluation. 

Each dataset includes a corresponding `.csv` file located in `root_dir/datasets/labels`, which contains all the labels for the images. The headers for each file are as follows:

-  Name
- File path
- Identity
- Gender code (1 for male, -1 for female)
- Gender
- Age
- Ethnicity
- Date of birth
- Emotion code (0 to 9)
- 9 emotions (Neutral, Anger, Scream, Contempt, Disgust, Fear, Happiness, Sadness, Surprise)
- Scarf
- Glasses
- Beard
- Headwear
- Face angle


- **Gender_code**: Male = 1, Female = -1  
- **Emotion_code**: 0 = Neutral, 1 = Anger, 2 = Scream, 3 = Contempt, 4 = Disgust, 5 = Fear, 6 = Happy, 7 = Sadness, 8 = Surprise  
- **Race_code**: Not yet defined  
- **Angle**: 90 = Right profile, 45 = 45° to the left, 0 = Frontal, -45 = 45° to the right, -90 = Left profile  

If a label is unavailable for any image, leave the field as an empty string.

You must add the labels.csv as indicated as follows:
```plaintext
📂 deid-toolkit
├── 📁 root_dir               # Main project directory
│   ├── 📂 datasets           # Datasets directory
│   │   ├── 📂 aligned        # Aligned images for processing
│   │   │   ├── 🖼️ img1.jpg   # Example image 1
│   │   │   ├── 🖼️ img2.jpg   # Example image 2
│   │   │   ├── ...
│   │   ├── 📂 labels        # Aligned images for processing
│   │   │   ├── 
```

### 3 Dataset integrated
To verify that the dataset has been correctly integrated, run the toolkit, type `select datasets`, and you will be able to select it.