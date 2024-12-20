# Techniques
Each technique is presented in the form of a script or a set of scripts that are independent of the
toolkit. 

## Table of contents
1. Basic usage 
2. How to integrate a dataset

## Basic usage
1) select datasets
2) select techniques


## How to add an implemented technique
Add a Python script technique_name.py in root_dir/techniques. This script must be callable with:
`python technique_name.py dataset_path dataset_save_path`

If the new technique needs an environment different from the toolkit, put a file technique_name.yml in the directory root_dir/environments

> [!IMPORTANT]
> The environment should be named using the same technique name, so the toolkit can automatically recognize it. If the environment name differs, you will need to manually configure it in the config.ini file. For more details on how to map environments with different names to techniques, please refer to the documentation. <--- add link there
<!-- TODO: add a link to explain how  config.ini manage custom envs -->
## How to add a not implmented technique
If have the deidentified images already and it's aligned version, you can evaluate them without integrating the new technique into the toolkit. 

The steps to do that are the following: 
1) add the aligned dataset into `.root_dir/datasets/aligned/` and set it a name like: _fooDataset_
2) add the deidentified images into `.root_dir/datasets/yourtechniqueName/fooDataset`
3) Create an empty file with the **same name** of the technique that you have been typed in the previous step.
4) Ready! Now you are able to select the dataset with your own technique.
