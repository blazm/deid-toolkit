
import utils as util
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn.metrics as metrics
import numpy as np
import itertools
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
def get_labeled_columns(dataframe: pd.DataFrame):
    available_labels = []
    for label in expression_labels:
        if label in dataframe.columns:
            if dataframe[label].notna().any():
                available_labels.append(label)
    return available_labels


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
    datasets = args.datasets
    evaluations = args.evaluations
    techniques = args.techniques
    path_to_save = args.path_save

    #combinations = itertools.product(evaluations, datasets, techniques)

    #num_plots = len(combinations)
    #cols = 2  
    #rows = int(np.ceil(num_plots / cols))  


    for i, evaluation in enumerate(evaluations):
        #plots settings
        num_plots = len(datasets)
        rows = int(np.sqrt(num_plots))
        cols = int(np.ceil(num_plots / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = np.atleast_1d(axes).ravel()

        for i, dataset in enumerate(datasets):
            ax = axes[i]
            for j, technique in enumerate(techniques):
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
                use_true_labels = True
                
                if df_filtered["Emotion_code"].isna().any():
                    print(f"(Warning) There is a missing information in {path_to_dataset_labels}: using model predicted value for aligned")
                    use_true_labels= False
                if use_true_labels: 
                    #if cannot find the emotions in labels  will the predicted values for the model in its aligned version
                    true_np = df_filtered["Emotion_code"].to_numpy()
                else:
                    true_np = df_filtered["aligned_predictions"].to_numpy()

                pred_np = df_pred["deidentified_predictions"].to_numpy()
                

                #get the emotions available for that model and dataset()
                original_labels = get_labeled_columns(df_original)# get the dataset emotions
                model_labels = df_pred["Emotion_code"].map(labels_map).tolist()# get the available emotions
                common_emotions = list(set(original_labels) & set(model_labels))

                valid_arr = generate_conf_matrix(true_np, pred_np, common_emotions)

                ax.set_aspect(1)
                res = ax.imshow(np.array(valid_arr), cmap=plt.cm.jet, 
                        interpolation='nearest') # , vmax=66 TODO: change this, 66 is for rafd
                width, height = np.array(valid_arr).shape
                for x in range(width):
                    for y in range(height):
                        ax.annotate(str(valid_arr[x][y]), xy=(y, x), 
                                    horizontalalignment='center',
                                    verticalalignment='center')
                alphabet = [i[0:3].upper() for i in expression_labels]
                plt.xticks(range(width), alphabet[:width])
                plt.yticks(range(height), alphabet[:height])
        to_save_pdf = os.path.join(path_to_save, f"confussion_matrix.pdf")
        plt.savefig(to_save_pdf, dpi=600)
        plt.close()
        print(f"Confusion Matrix saved in: {to_save_pdf}")

if __name__ == "__main__":
    args = util.read_args()
    main(args)    

