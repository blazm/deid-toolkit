
import utils as util
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.metrics as metrics
import numpy as np

colors = [
          '#111110',
          '#11111f', 
          '#222220',
          '#22222f', 
          '#333330',
          '#33333f', 
          '#334433',
          '#778877', 
          '#ff0000',
          '#ff8800', 
          '#008844',
          '#44ff44',
          '#0000ff',
          '#0088ff',
          '#ff00ff',
          '#ff88ff',
          '#888800',
          '#44aa22',
          '#111110',
          '#11111f', 
          '#222220',
          '#22222f', 
          '#333330',
          '#33333f', 
          '#334433',
          '#778877', 
          '#ff0000',
          '#ff8800', 
          '#008844',
          '#44ff44',
          '#0000ff',
          '#0088ff',
          '#ff00ff',
          '#ff88ff',
          '#888800',
          '#44aa22',]

def compute_eer(fpr,tpr,thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]

def main(args):
    dataset = args.dataset
    evaluation = args.evaluation
    technique = args.technique
    path_to_save = args.path_save

    path_to_csv =  os.path.abspath(os.path.join(util.RESULTS_DIR, f"{evaluation}_{dataset}_{technique}.csv"))
    df = pd.read_csv(path_to_csv)

    plt.figure(0)
    plt.title(f"Receiver Operating Characteristic ({evaluation}) ")
    ax_markers = []
    eps_symbols = []
    
 
    colors = ['#2D58DA', '#2D3CDA', '#2817E7', '#7213CC', '#B90FB9', '#E732C6', '#ED4E9B', '#F26A6F', '#F88B72', '#FD9F7A']
    markers = ['.', '+', '*', 'x', 'o', 's', 'p', 'D']
    plt.rc('legend', fontsize=3) 


    label ='{} - {}'.format(dataset, technique)
    ground_truth_binary_labels = df["ground_truth"].to_numpy()
    predicted_scores = df["cossim"].to_numpy()
    fpr, tpr, threshold = metrics.roc_curve(ground_truth_binary_labels, predicted_scores)
    roc_auc = metrics.auc(fpr, tpr)
    roc_eer, _ = compute_eer(fpr, tpr, threshold)
    plt.plot(fpr, tpr, 'b', label = label + "; AUC=%0.2f, EER=%0.2f" % (roc_auc, roc_eer),
        color=colors[0],
        linewidth=0.75,
        markersize=1)

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig(path_to_save, dpi=600)
    plt.close()

if __name__ == "__main__":
    args = util.read_args()
    main(args)    
## TODO: CHOOSE DATASET AND TECHNIQUE (USER SELECTION)