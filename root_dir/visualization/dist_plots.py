import utils as util
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def main(args):
    datasets = args.datasets
    evaluations = args.evaluations
    techniques = args.techniques
    path_to_save = args.path_save

    kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)

    for evaluation in evaluations: 
        # Plot settings: rows for techniques, cols for datasets
        rows = len(techniques)
        cols = len(datasets)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 15), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        
        for i, dataset in enumerate(datasets):
            for j, technique in enumerate(techniques):
                idx = j * len(datasets) + i  # Flattening index
                ax = axes[idx]
                ax.set_title(f"{dataset} - {technique}", fontsize=10)
                
                path_to_csv = os.path.abspath(os.path.join(util.RESULTS_DIR, f"{evaluation}_{dataset}_{technique}.csv"))
                if not os.path.exists(path_to_csv):
                    continue  # Skip if the file doesn't exist    

                df = pd.read_csv(path_to_csv)
                # Read information from dataset                
                ground_truth_binary_labels = df["ground_truth"].to_numpy()
                predicted_scores = df["cossim"].to_numpy()

                genuine_probabilities = predicted_scores[ground_truth_binary_labels == 1]
                impostor_probabilities = predicted_scores[ground_truth_binary_labels == 0]

                # Plot histograms
                ax.hist(genuine_probabilities, color='blue', **kwargs)
                ax.hist(impostor_probabilities, color='red', **kwargs)
                ax.set_xlim(0, 1)  # X-axis shared across plots

        # Add layout adjustments
        plt.tight_layout()
        to_save_pdf = os.path.join(path_to_save, f"{evaluation}_dist.pdf")
        print(f"Plot dist for {evaluation} saved in: {to_save_pdf}")
        plt.savefig(to_save_pdf, dpi=600)
        plt.close()


if __name__ == "__main__":
    args = util.read_args()
    main(args)    

