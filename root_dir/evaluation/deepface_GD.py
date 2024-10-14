from deepface import DeepFace
import os
import argparse
import utils as util



def main():
    output_score = util.MetricsBuilder()
    args = util.read_args()
    aligned_dataset_path = args.aligned_path
    deidentified__dataset_path  = args.deidentified_path
    path_to_save = args.save_path
    files = os.listdir(aligned_dataset_path)
    output_score_file = util.get_output_filename("deepface", aligned_dataset_path, deidentified__dataset_path)
    f = open(output_score_file, 'w')

    dataset_name = util.get_dataset_name_from_path(aligned_dataset_path)
    technique_name = util.get_technique_name_from_path(deidentified__dataset_path)
    metrics_df= util.Metrics(name_evaluation="deepface", 
                              name_dataset=dataset_name,
                              name_technique=technique_name,
                              name_score="isMatch")
    
    device = 'cuda' if True else 'cpu'


    samples:int = len(files)
    succeses:int = 0     
    for file in files: 
        aligned_img_path = os.path.join(aligned_dataset_path, file)
        deidentified_img_path = os.path.join(deidentified__dataset_path, file)
        if not os.path.exists(aligned_img_path):
            print(f"{aligned_dataset_path} does not exist")
            continue
        if not os.path.exists(deidentified_img_path):
            print(f"{deidentified__dataset_path} does not exist")
            continue
        #run the predicctions
        pred_aligned = DeepFace.analyze(img_path = aligned_img_path, actions = ['gender'],enforce_detection=False)
        pred_deidentified = DeepFace.analyze(img_path = deidentified_img_path, actions = ['gender'],enforce_detection=False)
        #Log the result
        #f.writelines(f"{emotion_aligned}, {emotion_deidentified},{True if emotion_aligned == emotion_deidentified else False}")
        output_score.add_output_message(pred_aligned)
        output_score.add_output_message(pred_deidentified)
        gender_aligned = pred_aligned[0].get("dominant_gender", [])#.get("dominant_gender", "-")
        gender_deidentified = pred_deidentified[0].get("dominant_gender", [])#.get("dominant_gender", "--")
        

        #Increase the succeses if are equal
        if gender_aligned == gender_deidentified: 
            succeses+=1
        is_match = 1 if gender_aligned == gender_deidentified else 0
        metrics_df.add_score(aligned_img_path, deidentified_img_path, metric_result=is_match)
    f.close()
    accuracy= 0
    accuracy = (succeses / samples)*100
    metrics_df.save_to_csv(path_to_save)
    return output_score.add_metric("deepface", "accuracy", "{:1.2f}%".format(accuracy))

if __name__ == "__main__":
    #main()
    result, output, _  =util.with_no_prints(main)
    print(result.build())
