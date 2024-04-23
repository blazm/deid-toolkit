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

## ToolKit Command Examples

### general commands
root - see currently set root directory
set root - set root directory
serve - run webserver to see the generated results
set serve root - set results directory for serving results

load config "filename.ini"
save config "filename.ini"

### listing commands for displaying implemented methods, * is a wildcard to run all
list datasets
list techniques
list evaluation
list visuals
list *

### preprocessing commands (once the datasets were selected)
preprocess images
preprocess labels
preprocess *  

### dataset|technique|evaluation selection
select datasets
select techniques
select evaluation
select *
current selection

### running the processing (with feedback on the progress)
run preprocess
run techniques
run evaluation
run *