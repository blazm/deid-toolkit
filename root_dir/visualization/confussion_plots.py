
import utils as util
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.metrics as metrics
import numpy as np

expression_labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

def generate_conf_matrix(true, pred, expression_labels):
   
    s = len(expression_labels)
    matrix = np.zeros((s, s))

    for t, p in zip(true, pred):
        #print(t, p)
        x = expression_labels[t]
        y = expression_labels[p]
        matrix[t,p] += 1

    return matrix


def main(args):
    path_scores = args.path_scores
    path_to_save = args.path_save
    evaluation_name = os.path.basename(path_scores)

    df = pd.read_csv(path_scores)
    df = df.drop(columns=['aligned_path', 'deidentified_path'])
    grouped = df.groupby(['dataset', 'technique'])  

    plt.figure(0)
    plt.title(f"Receiver Operating Characteristic ({evaluation_name}) ")
    
if __name__ == "__main__":
    #args = util.read_args()
    #main(args)    
    pass