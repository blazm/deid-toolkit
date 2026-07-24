"""DeepFace-based facial expression classification evaluation."""
import os
from deepface import DeepFace
import utils as util
from tqdm import tqdm

# Expression codes matching toolkit convention:
# 0=Neutral, 1=Anger, 2=Scream, 3=Contempt, 4=Disgust, 5=Fear, 6=Happy, 7=Sadness, 8=Surprise
labels_map = {
    "neutral": 0, "angry": 1, "surprise": 8,
    "disgust": 4, "fearful": 5, "happy": 6, "sad": 7,
}


def main():
    args = util.read_args()
    aligned_path = args.aligned_path
    deid_path = args.deidentified_path
    path_to_save = args.save_path
    path_to_log = args.dir_to_log

    files = os.listdir(aligned_path)
    ds = util.get_dataset_name_from_path(aligned_path)
    tech = util.get_technique_name_from_path(deid_path)
    metrics_df = util.Metrics(name_score="isMatch")

    for i, file in enumerate(tqdm(files, total=len(files), desc=f"deepface_expression | {ds}-{tech}")):
        if util._TEST_SINGLE and i > 0:
            break
        path_a = os.path.join(aligned_path, file)
        path_d = os.path.join(deid_path, file)
        if not os.path.exists(path_a):
            util.log(os.path.join(path_to_log, "deepface_expression.txt"), f"({ds}) Missing: {path_a}")
            continue
        if not os.path.exists(path_d):
            util.log(os.path.join(path_to_log, "deepface_expression.txt"), f"({tech}) Missing: {path_d}")
            continue

        pred_a = DeepFace.analyze(img_path=path_a, actions=["emotion"], detector_backend="skip")
        pred_d = DeepFace.analyze(img_path=path_d, actions=["emotion"], detector_backend="skip")

        ea = pred_a[0].get("dominant_emotion")
        ed = pred_d[0].get("dominant_emotion")
        is_match = 1 if ea == ed else 0
        metrics_df.add_score(file, is_match)
        metrics_df.add_column_value("aligned_predictions", labels_map.get(ea.lower(), -1))
        metrics_df.add_column_value("deidentified_predictions", labels_map.get(ed.lower(), -1))

    metrics_df.save_to_csv(path_to_save)
    print(f"deepface_expression saved into {path_to_save}")


if __name__ == "__main__":
    main()
