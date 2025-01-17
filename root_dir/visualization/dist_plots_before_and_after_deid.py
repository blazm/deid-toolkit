import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import combinations
from scipy.stats import gaussian_kde
import utils as util
import os
from itertools import product
from tqdm import tqdm




def main(args): 
    datasets = args.datasets
    evaluations = args.evaluations
    techniques = args.techniques
    path_to_save = args.path_save

    #plot features
    kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)

    for evaluation in evaluations: 
        rows = len(techniques)
        cols = len(datasets)
        fig, axes = plt.subplots(rows, cols, figsize=(7, 15), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for idx, (dataset, technique) in enumerate(product(datasets, techniques)):
            ax = axes[idx]

            path_to_csv = os.path.abspath(os.path.join(util.RESULTS_DIR, f"{evaluation}_{dataset}_{technique}.csv"))
            if not os.path.exists(path_to_csv):
                continue  # Skip if the file doesn't exist 
            df = pd.read_csv(path_to_csv)
             # Read information from dataset                
            ground_truth_binary_labels = df["ground_truth"].to_numpy()
            predicted_scores = df["cossim"].to_numpy()

            genuine_scores_deid = predicted_scores[ground_truth_binary_labels == 1]
            impostor_scores_deid = predicted_scores[ground_truth_binary_labels == 0]
            genuine_kde_deid = gaussian_kde(genuine_scores_deid)
            impostor_kde_deid = gaussian_kde(impostor_scores_deid)
            if "cossim_originals" in df.columns:
                print(f"(Note) {os.path.basename(path_to_csv)} doesn't contain 'cossim_originals' scores. To fix this please check the selected evaluation method ")
                predicted_scores_originals = df["cossim_originals"].to_numpy()
                genuine_scores_original = predicted_scores_originals[ground_truth_binary_labels == 1]
                genuine_kde_original = gaussian_kde(genuine_scores_original)

                x_values = np.linspace(min(min(genuine_scores_deid), min(impostor_scores_deid), min(genuine_scores_original)),
                                    max(max(genuine_scores_deid), max(impostor_scores_deid), max(genuine_scores_original)), 1000)
            else: 
                x_values = np.linspace(
                            min(min(genuine_scores_deid), min(impostor_scores_deid)),
                            max(max(genuine_scores_deid), max(impostor_scores_deid)), 
                            1000
                        )
            genuine_density_deid = genuine_kde_deid(x_values)
            impostor_density_deid = impostor_kde_deid(x_values)
            if "cossim_originals" in df.columns:
                genuine_density_original = genuine_kde_original(x_values)
            
            #TODO: get the original genuines pairs.
            # Kernel density estimation (KDE) for all distributions

            # Create a range of x values covering all distributions

            # Calculate the densities for all distributions

            # Find the minimum of the densities for overlap (between genuine and impostor after de-identification)
            overlap_density = np.minimum(genuine_density_deid, impostor_density_deid)
            # Plot the distributions
            ax.fill_between(x_values, overlap_density, color='purple', alpha=0.5, label='Overlap Area (After De-Identification)')
            ax.plot(x_values, genuine_density_deid, color='blue', label='Genuine KDE (After De-Identification)', alpha=0.7)
            ax.plot(x_values, impostor_density_deid, color='red', label='Impostor KDE (After De-Identification)', alpha=0.7)
            if "cossim_originals" in df.columns:
                ax.plot(x_values, genuine_density_original, color='green', linestyle='--', label='Genuine KDE (Original)', alpha=0.7)

            # Labels and title
            ax.set_title(f"{evaluation}-{dataset}-{technique}")
            ax.set_xlabel('Similarity Score')
            ax.set_ylabel('Density')
            ax.legend(fontsize=7, title_fontsize=11)
    to_save_pdf = os.path.join(path_to_save, f"before_after_dist_plots.pdf")
    plt.title('Comparison of Genuine and Impostor Distributions Before and After De-Identification')
    plt.subplots_adjust( hspace=0.5)
    plt.savefig(to_save_pdf, dpi=600)
    plt.close()
    print(f"scores table saved in: {to_save_pdf}")




            






if __name__ == "__main__":
    args = util.read_args()
    main(args)    

