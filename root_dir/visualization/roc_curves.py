
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
    datasets = args.datasets
    evaluations = args.evaluations
    techniques = args.techniques
    path_to_save = args.path_save

    for evaluation in evaluations:
        # Crear un PDF por evaluación
        num_plots = len(datasets)  # Número total de gráficos (uno por dataset)
        rows = int(np.sqrt(num_plots))  # Filas de la cuadrícula
        cols = int(np.ceil(num_plots / rows))  # Columnas de la cuadrícula

        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

        # Asegurarse de que axes sea un array, incluso si solo hay un gráfico
        axes = np.atleast_1d(axes).ravel()  # Aplanar para iterar fácilmente

        #mean_fpr = np.linspace(0, 1, 100)

        for i, dataset in enumerate(datasets):
            ax = axes[i]  # Selecciona el subplot correspondiente al dataset
            ax.set_title(f"ROC ({evaluation}): {dataset}")
            
            # Graficar una línea por cada técnica en el mismo gráfico
            for j, technique in enumerate(techniques):
                path_to_csv = os.path.abspath(os.path.join(util.RESULTS_DIR, f"{evaluation}_{dataset}_{technique}.csv"))
                
                if not os.path.exists(path_to_csv):
                    continue  # Si el archivo no existe, pasar a la siguiente técnica
                
                df = pd.read_csv(path_to_csv)
                ground_truth_binary_labels = df["ground_truth"].to_numpy()
                predicted_scores = df["cossim"].to_numpy()
                fpr, tpr, threshold = metrics.roc_curve(ground_truth_binary_labels, predicted_scores)
                roc_auc = metrics.auc(fpr, tpr)
                roc_eer, _ = compute_eer(fpr, tpr, threshold)
                
                label = f'{technique} (AUC={roc_auc:.2f}, EER={roc_eer:.2f})'
                color = plt.cm.get_cmap('tab10')(j)  # Usar un color diferente para cada técnica
                
                ax.plot(fpr, tpr, label=label, color=color, linewidth=0.75)
                ax.plot([0, 1], [0, 1], 'r--')  # Línea diagonal (aleatorio)

            # Configurar límites, etiquetas y leyenda
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_ylabel('True Positive Rate')
            ax.set_xlabel('False Positive Rate')
            ax.legend(loc='lower right', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(path_to_save, f"{evaluation}_roc_curves.pdf"), dpi=600)
        plt.close()

if __name__ == "__main__":
    args = util.read_args()
    main(args)    
## TODO: CHOOSE DATASET AND TECHNIQUE (USER SELECTION)