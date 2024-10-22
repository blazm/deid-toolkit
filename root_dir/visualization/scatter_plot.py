
import utils as util
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def main(args):
    path_scores = args.path_scores
    path_save = args.path_save
    evaluation_name = os.path.basename(path_scores)

    df = pd.read_csv(path_scores)
    df = df.drop(columns=['aligned_path', 'deidentified_path'])
    grouped = df.groupby(['dataset', 'technique'])  

    plt.figure(0)
    plt.title(f"Receiver Operating Characteristic ({evaluation_name}) ")
    #for ((dataset_name, technique_name), group), color in zip(grouped, colors):
    #    pass 

    
    


if __name__ == "__main__":
    args =util.read_args()
    main(args)