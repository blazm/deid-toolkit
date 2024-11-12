
import utils as util
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.metrics as metrics
import numpy as np
import itertools
import matplotlib.colors as mcolors
expression_labels = ["Neutral","Anger","Scream","Contempt","Disgust","Fear","Happy","Sadness","Surprise"]
labels_map= {0:"Neutral", 
             1:"Anger", 
             2:"Scream", 
             3:"Contempt", 
             4:"Disgust",
             5:"Fear",
             6:"Happy",
             7:"Sadness",
             8:"Surprise"} 
#we should be consistent with the names and number of the labels for each evaluation method
cmap = plt.cm.jet
cmap_with_grey = mcolors.ListedColormap(['lightgrey'] + [cmap(i) for i in range(cmap.N)])

def generate_conf_matrix(true, pred, common_emotions):
    emotion_indices = {value: key for key, value in labels_map.items()}

    matrix = np.zeros((len(expression_labels), len(expression_labels)))

    for t, p in zip(true, pred):
        emotion_true = labels_map.get(t)
        emotion_pred = labels_map.get(p)
        x = emotion_indices[emotion_true]
        y = emotion_indices[emotion_pred]
        matrix[x, y] += 1
    return matrix

    

def main(args):
    datasets = args.datasets
    evaluations = args.evaluations
    techniques = args.techniques
    path_to_save = args.path_save

    for i, evaluation in enumerate(evaluations):
        #plots settings
        rows = len(techniques)
        cols = len(datasets)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5)  
        axes = np.atleast_1d(axes).ravel()

        for i, dataset in enumerate(datasets):
            for j, technique in enumerate(techniques):
                idx = j * len(datasets) + i  # Flattening index
                ax = axes[idx]
                ax.set_title(f"Confusion matrix: {dataset}-{technique}",fontsize=12)
                path_to_dataset_labels = os.path.abspath(os.path.join(util.LABELS_DIR, f"{dataset}_labels.csv"))
                path_to_scores = os.path.abspath(os.path.join(util.RESULTS_DIR, f"{evaluation}_{dataset}_{technique}.csv"))
                if not os.path.exists(path_to_scores):
                    print(f"{path_to_scores} doesn't exist - (skip)")    
                    continue  # Skip if the file doesn't exist    
                if not os.path.exists(path_to_dataset_labels):
                    print(f"{path_to_dataset_labels} doesn't exist - (skip)")
                    continue  # Skip if the file doesn't exist  
                df_pred = pd.read_csv(path_to_scores) #
                df_original= pd.read_csv(path_to_dataset_labels)

                 #prevent the missing images after deidentification
                df_filtered = df_original[df_original['Name'].isin(df_pred['img'])]
                
                if df_filtered["Emotion_code"].isna().any():
                    #if cannot find the emotions in labels  will the predicted values for the model in its aligned version
                    print(f"(Warning) There is a missing information in {path_to_dataset_labels}: using model predicted value for aligned")
                    true_np = df_pred["aligned_predictions"].to_numpy()
                else:
                    #otherwise, use the original
                    true_np = df_filtered["Emotion_code"].to_numpy()
                pred_np = df_pred["deidentified_predictions"].to_numpy()

                #get the emotions available for that model and dataset()
                model_labels = list(set(df_pred["deidentified_predictions"].map(labels_map).tolist()))# get the available emotions
                common_emotions = list(set(expression_labels) & set(model_labels))

                conf_matrix = generate_conf_matrix(true_np, pred_np, common_emotions)
                conf_matrix_masked = np.where(
                        np.array([[x in common_emotions and y in common_emotions for y in expression_labels] for x in expression_labels]),
                        conf_matrix,
                        np.nan
                    )
                ax.set_aspect(1)
                im = ax.imshow(conf_matrix_masked, cmap=cmap_with_grey, interpolation='nearest')
                #ax.imshow(conf_matrix, cmap=cmap, interpolation='nearest')
                for x in range(conf_matrix.shape[0]):
                    for y in range(conf_matrix.shape[1]):
                        value = conf_matrix[x, y]
                        color = 'black' if expression_labels[y] in common_emotions and expression_labels[x] in common_emotions else 'grey'
                        ax.text(y, x, str(value), ha='center', va='center', color=color)


                ax.set_xticks(range(len(expression_labels)), expression_labels, rotation=90)
                ax.set_yticks(range(len(expression_labels)), expression_labels)

        to_save_pdf = os.path.join(path_to_save, f"confusion_matrix_{evaluation}.pdf")
        plt.savefig(to_save_pdf, dpi=600)
        plt.close()
        print(f"Confusion Matrix saved in: {to_save_pdf}")

if __name__ == "__main__":
    args = util.read_args()
    main(args)    

