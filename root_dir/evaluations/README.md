# Evaluation
This component evaluates the results. It checks both the aligned dataset and the de-identified dataset, executes each metric and generates logs for tracking. It also includes utilities to help the integration of the future evaluation methods. Additionally, the scores will be stored in a CSV file.

## Table of contents
1. Basic usage 
2. How the results are stored
2. Module structure
    * utils 
3. How to integrate an evaluation metric

## Basic usage

## Module structure
The evaluation follows the next structure:

```plaintext
📂 deid-toolkit
├── 📁 root_dir/               
│   ├── 📂 evaluation/           
│   │   ├── 📂 data_utility/              
│   │   ├── 📂 identity_verification/              
│   │   ├── 📂 image_quality/              
│   │   ├── 📂 tmp/              
│   │   ├── 📂 utils/              
│   │   │   ├── ____init___.py      
│   │   ├── evaluation1.py           
│   │   ├── evaluation2.py           
│   │   ├── evaluation3.py           
│   │   ├── ...         
```

### identity_verification directory
This directory handles identity verification evaluations and ROC (Receiver Operating Characteristic) curve generation. The models used for identity verification are:
* VGG-Face
* AdaFace
* SwinFace

To perform these evaluations, you need to generate image pairs as described below:
* Genuine pairs: Two images with matching identities.
* Impostor pairs: Two images with non-matching identities.

> [!NOTE]
> There are two versions of VGG-Face and AdaFace. One version stores features in the tmp folder (optimized). If the features for an image are already stored, the toolkit will load them and use them instead of re-extracting the features.


### data_utility directory 
The data_utility directory provides utility scripts and functions essentially comparing classifiers, before and after deidentification. For this we have integrated two kinds of models:
* Facial Expression Models  
    * DAN (Deep Action Units Network)
    * HSEmotion
* Gender detection Models
    * ResNet18
    * DeepFace

Some models can perform different tasks. For example, a single model can handle both gender classification and facial expression classification. Instead of creating multiple environments with the same dependencies, we recommend mapping an environment manually to optimize resource usage. See **adaface and adaface_optimized example in config.ini <------------------TODO >

> [!TIP]
> Create distinct scripts, such as evaluation_for_gd.py for gender detection and evaluation_for_ex.py for expression classification, to keep tasks organized.

> [!CAUTION]
> Since these are classification models, each **model may format its predictions differently**, such as using labels like "woman" and "man" or "male" and "female." Additionally, expression indices may vary between models. **It's necessary to map the model outputs to the expected labels** or indices in the `labels.csv` file (see **Datasets Labels** section for more details).  
For example, if a model's output for gender detection is "man" or "woman," map the results as follows:  
`labels_map = {"Man": 1, "Woman": -1}`



### image_quality
This folder contains the scripts for evaluating the image quality based on various metrics such as: 
* FID (Fréchet Inception Distance)
* LPIPS (Learned Perceptual Image Patch Similarity)
* MSE (Mean Squared Error)
* SSIM (Structural Similarity Index)

> [!NOTE]
> identity_verification | data_utility | image_quality directories purpose is to organize the evaluation methods. However, they are not required to strictly follow this structure for the toolkit to function properly, as long as the scripts in the `evaluation/` folder can correctly call the functions.
### utils

* explain the logs 
    
* explain the temp files (features)
    * you have to manually remove the temp directory if you want to extract the features again.
* explain we need to map the classification features to our labels
    * example about map
* Explain metrics df (to stored the results as a standarized way so that the toolkit can undersdand the results)
    * explain how to add a new information in the same dataset.


## How to integrate an evaluation metric
> [!TIP]
> **We highly recommend** relying on the functions available in the `utils` when integrating a new evaluation method.
> [!TIP]
> add a tip here about the use of temp file to save features 
* How the scripts must be callabe with

* How to create an environment and how to use the same environment for several python files
    * refer information about map 
* using logs.


