# deid-toolkit
An attempt to develop a toolkit for running and evaluating privacy preserving techniques in facial biometrics

## Toolkit description



![Architecture](https://github.com/blazm/deid-toolkit/blob/main/assets/architecture.png?raw=true)

## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Examples](#examples)
7. [Project Structure](#project-structure)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

-- 
## Features
* Command-line interface (something like https://docs.python.org/3.8/library/cmd.html):
    *   easy to use (simple commands + names to run experiments)
    * easy to generate results
    * good look & feel
    * responsive logging (showing % of performed actions)
    * helpful tips (where results are saved, how can be visualised, etc)
* Handling of multiple separated virtual environments
    * running the models in subshells, while reporting progress to main interface
* Configuration parameters in text format via .ini file (loading & saving of different configurations within command line)
* Checking and handling of datasets
    * Raw datasets path (simple names)
    * Cropped & Aligned path (dataset preparation/standardization)
* Saving intermediate results for each phase (for each dataset, for each of the models)
* Saving final results (final deidentified images and dataset evaluation scores / plots)
* Handling and storing models / binary files for existing techniques
<!-- TO DO: look and check remove needed -->
    *   Pretrained models directory 

--

## Prerequisites
- **Operating System:** Linux ? 
- **Python:** 3.9+
- **Additional Dependencies:**  Conda, Mamba 

## Installation

1. Clone the project: 
   ```sh
   git clone https://github.com/blazm/deid-toolkit
   ```
<!-- TODO: Where to download this zips-->
2. Get `techniques.zip` and `aligned.zip` (and `original.zip` if wanted) and extract them with unzip:
   ```sh
   unzip techniques.zip -d root_dir
   unzip evaluation.zip -d root_dir
   unzip visualization.zip -d root_dir
   unzip aligned.zip -d root_dir/datasets
   unzip original.zip -d root_dir/datasets
   ```

3. Create the toolkit environment:
   ```sh
   conda env create -f toolkit.yml
   ```
4. In `deid_shell.py`, change the `conda_sh_path` constant with the correct path to the conda.sh file on your machine.

## Usage
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
- `run visualize`
- `run *`

<!-- TODO: add link for visualizations-->
> [!NOTE]
> There is no selection for visualization methods, please refer to [visualization]() to discover more details.

> [!TIP]
> Your selection is stored in config.ini file: Which means you don't have to select again _dataset|technique|evaluation|_ if you want to run the same selected _dataset|technique|evaluation|_

--

## Toolkit components
### Datasets
<!-- TODO: add link in the word "integrate" to learn how-->
This module manages the datasets required for de-identification. It’s the first part of the pipeline. The toolkit is able to integrate (<-----) additional facial images datasets. Moreover, the datasets won’t be included in the toolkit because some of them have different licensing constraints.

<!-- TODO: add link to more details in README_DATASETS-->


### Preprocess
### Environments
### Techniques
### Evaluation
### Visualization



