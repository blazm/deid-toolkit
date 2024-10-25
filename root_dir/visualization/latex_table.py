import pandas as pd
import os
import utils as util
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    datasets = args.datasets
    evaluations = args.evaluations
    techniques = args.techniques
    path_to_save = args.path_save

    num_plots = len(techniques)
    cols = 2  
    rows = int(np.ceil(num_plots / cols))  

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10)) 
    axes = np.atleast_1d(axes).ravel()


    for i, technique in enumerate(techniques):
        rows = []  # Store rows for the DataFrame
        ax = axes[i]
        ax.set_title(f"{technique}", fontsize=10)

        for evaluation in evaluations:
            mean_std_evaluations = [evaluation.replace("_", "\_")]  # First column is the dataset name

            for dataset in datasets:
                # Path to the CSV file
                path_to_csv = os.path.abspath(os.path.join(util.RESULTS_DIR, f"{evaluation}_{dataset}_{technique}.csv"))

                if not os.path.exists(path_to_csv):
                    mean_std_evaluations.append("Nan")
                    continue  # Skip if the file doesn't exist
                
                # Read CSV and calculate mean and std deviation
                df = pd.read_csv(path_to_csv)
                df_scores = df.iloc[:, 1]  # Assuming the second column contains the scores
                mean_std_evaluations.append(f"${df_scores.mean():.2f} \pm {df_scores.std():.2f}$")

            rows.append(mean_std_evaluations)  # Append the row for this dataset

        # Create a DataFrame with the collected rows
        datasets_columns = [d.replace("_", "\_") for d in datasets]
        table = pd.DataFrame(rows, columns=["evaluation"] + datasets_columns)

        # Save the LaTeX table to a text file
        latex_filename = os.path.join(path_to_save, f"table_{technique}.txt")
        with open(latex_filename, 'w') as f:
            f.write(table.to_latex(index=False))
        print(f"Saved LaTeX for {technique} in {latex_filename}")
        # add table to the subplot
        ax.table(cellText=table.values, colLabels=table.columns, cellLoc='center', loc='center')
        ax.axis('tight')
        ax.axis('off')

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')  # Turn off the unused axes

    to_save_pdf = os.path.join(path_to_save, f"table_scores.pdf")
    plt.savefig(to_save_pdf, dpi=600)
    plt.close()
    print(f"scores table saved in: {to_save_pdf}")


if __name__ == "__main__":
    args = util.read_args()
    main(args)
