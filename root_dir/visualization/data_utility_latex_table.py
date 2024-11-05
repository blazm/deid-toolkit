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

    # configure the plot
    num_plots = len(datasets)
    cols = 1
    rows = int(np.ceil(num_plots/ cols))

    fix, axes = plt.subplots(rows, cols, figsize=(15,10))
    axes = np.atleast_1d(axes).ravel()

    for i, dataset in enumerate(datasets): 
        #one table per dataset
        rows = [] #store the rows, rows will be the techniques, and the columns will be the evaluation methods
        ax = axes[i]
        ax.set_title(f"Data utility (gender and expression classification accuracy) for {dataset}", fontsize=10)
        for technique in techniques:
            accuracy_columns = [technique.replace("_", "\_")]  # first colum will the the techniques

            for evaluation in evaluations: 
                #for each column in the row, open the csv file
                path_to_csv = os.path.abspath(os.path.join(util.RESULTS_DIR, f"{evaluation}_{dataset}_{technique}.csv"))
                if not os.path.exists(path_to_csv):
                    #skip because couldn't find the metrics to compute accuracy
                    accuracy_columns.append("Nan")
                    continue 
                df = pd.read_csv(path_to_csv)
                total_values = len(df["isMatch"]) # should be the second colum
                successes = df["isMatch"].sum()                
                accuracy = successes / total_values
                accuracy_columns.append(f"{accuracy:.2f}")
            rows.append(accuracy_columns)
        #Create a Dataframe with the collected rows
        columns_headers = [ev.replace("_", "\_") for ev in evaluations]
        table = pd.DataFrame(rows, columns=["techniques"] + columns_headers)

         # Save the LaTeX table to a text file
        latex_filename = os.path.join(path_to_save, f"table_{dataset}.txt")
        with open(latex_filename, 'w') as f:
            f.write(table.to_latex(index=False))
        print(f"Saved LaTeX for {dataset} in {latex_filename}")
        ax.table(cellText=table.values, colLabels=table.columns, cellLoc='center', loc='center')
        ax.axis('tight')
        ax.axis('off')
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')  # Turn off the unused axes
    to_save_pdf = os.path.join(path_to_save, f"data_utility_tables.pdf")
    plt.savefig(to_save_pdf, dpi=600)
    plt.close()
    print(f"scores table saved in: {to_save_pdf}")

if __name__ == "__main__":
    args  = util.read_args( )
    main(args)