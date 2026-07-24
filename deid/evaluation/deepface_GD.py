"""DeepFace-based gender classification evaluation."""
import os
from deepface import DeepFace
import utils as util
from tqdm import tqdm

labels_map = {"Man": 1, "Woman": -1}


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

    for i, file in enumerate(tqdm(files, total=len(files), desc=f"deepface_gender | {ds}-{tech}")):
        if util._TEST_SINGLE and i > 0:
            break
        path_a = os.path.join(aligned_path, file)
        path_d = os.path.join(deid_path, file)
        if not os.path.exists(path_a):
            util.log(os.path.join(path_to_log, "deepface_gender.txt"), f"({ds}) Missing: {path_a}")
            continue
        if not os.path.exists(path_d):
            util.log(os.path.join(path_to_log, "deepface_gender.txt"), f"({tech}) Missing: {path_d}")
            continue

        pred_a = DeepFace.analyze(img_path=path_a, actions=["gender"], detector_backend="skip")
        pred_d = DeepFace.analyze(img_path=path_d, actions=["gender"], detector_backend="skip")

        ga = pred_a[0].get("dominant_gender")
        gd = pred_d[0].get("dominant_gender")
        is_match = 1 if ga == gd else 0
        metrics_df.add_score(file, is_match)
        metrics_df.add_column_value("aligned_predictions", labels_map.get(ga, 0))
        metrics_df.add_column_value("deidentified_predictions", labels_map.get(gd, 0))

    metrics_df.save_to_csv(path_to_save)
    print(f"deepface_gender saved into {path_to_save}")


if __name__ == "__main__":
    main()
