import utils as util
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.metrics as metrics
import numpy as np
from matplotlib.colors import hsv_to_rgb

def create_colors(datasets, techniques):
    colors = []
    # Create a color map (can be changed to 'plasma', 'viridis', etc.)
    cmap = plt.cm.plasma
    total_techniques = len(techniques)
    # For each technique, assign a unique color from the colormap
    for i, _ in enumerate(techniques):
        color = cmap(i / total_techniques)  # Distribute colors evenly
        colors.append(color)
    
    return colors    

def compute_eer(fpr, tpr, thresholds):
    """Returns equal error rate (EER) and the corresponding threshold."""
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]

def main(args):
    datasets = args.datasets
    evaluations = args.evaluations
    techniques = args.techniques
    path_to_save = args.path_save

    #the numbers of fold performs a robust and accurate curve
    n_folds = 10
    perm = None

    colors:list = create_colors(datasets, techniques)


    #evaluation could be differents models
    for evaluation in evaluations: 

        #plots settings
        num_plots = len(datasets)
        rows = int(np.sqrt(num_plots))
        cols = int(np.ceil(num_plots / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = np.atleast_1d(axes).ravel()
        
        #colors
        cmap = plt.cm.plasma
        i = 0    
        amt = -1

        for i, dataset in enumerate(datasets):
            ax = axes[i]
            ax.set_title(f"ROC ({evaluation}): {dataset}")
            
            
            for j, (technique, color) in enumerate(zip(techniques, colors)):
                
                path_to_csv = os.path.abspath(os.path.join(util.RESULTS_DIR, f"{evaluation}_{dataset}_{technique}.csv"))
                if not os.path.exists(path_to_csv):
                    continue  # Skip if the file doesn't exis
                
                #read information from dataset                
                df = pd.read_csv(path_to_csv)
                ground_truth_binary_labels = df["ground_truth"].to_numpy()
                predicted_scores = df["cossim"].to_numpy()

                tprs = []
                fprs = []
                aucs = []
                eers = []
                mean_fpr = np.linspace(0, 1, 100)

                ground_truth_labels = ground_truth_binary_labels.copy() # make a copy, to keep the original labels ordered
                step = predicted_scores.shape[0] / n_folds

                def unison_shuffled_copies(a, b, perm):
                    assert len(a) == len(b)
                    if perm is None:
                        perm = np.random.permutation(len(a))
                    #p = np.random.permutation(len(a))
                    #return a[p], b[p]
                    return a[perm], b[perm], perm
                
                predicted_probabilities, ground_truth_labels, perm = unison_shuffled_copies(predicted_scores, ground_truth_labels, perm)

                for k in range(1, n_folds+1):
                    begin = int((k-1)*step)
                    end  = int(k*step)
                    predicted_fold = predicted_probabilities[begin:end]
                    ground_truth_fold = ground_truth_labels[begin:end]

                    fpr_fold, tpr_fold, threshold_fold = metrics.roc_curve(ground_truth_fold, predicted_fold)
                    roc_auc_fold = metrics.auc(fpr_fold, tpr_fold)
                    roc_eer_fold, _ = compute_eer(fpr_fold, tpr_fold, threshold_fold)

                    tprs.append(np.interp(mean_fpr, fpr_fold, tpr_fold))
                    tprs[-1][0] = 0.0

                    aucs.append(roc_auc_fold)
                    eers.append(roc_eer_fold)

                # Store TPR and FPR for later analysis
                aucs = np.array(aucs)
                eers = np.array(eers)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0

                mean_auc = metrics.auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)

                # plot the data
                #label = technique + "; AUC=%0.2f $\pm$ %0.2f" % (mean_auc, std_auc),                
                #label = f'{technique} (AUC={mean_auc:.2f} ± {std_auc:.2f})'
                #ax.set_aspect('auto')
                ax.plot(mean_fpr, mean_tpr, color=color,
                label=technique + "; AUC=%0.2f $\pm$ %0.2f" % (mean_auc, std_auc),
                alpha=.8, linewidth=0.75, markersize=1)
                
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                #fill with std
                ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.1,
                    )

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.plot([0, 1], [0, 1], color='#777777', linestyle='dashed', linewidth=0.5)
            ax.set_ylabel('Verification rate (VER)')
            ax.set_xlabel('False acceptance rate (FAR)')
            ax.legend(loc = 'lower right', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(path_to_save, f"{evaluation}_roc_curves.pdf"), dpi=600)
        plt.close()

if __name__ == "__main__":
    args = util.read_args()
    main(args)    

