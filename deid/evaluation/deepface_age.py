"""DeepFace-based age classification evaluation."""
import os
from deepface import DeepFace
import utils as util
from tqdm import tqdm


def age_bin(age):
    """Map numeric age to bucket label."""
    if age < 13:
        return "child"
    elif age < 19:
        return "teenager"
    elif age < 36:
        return "adult"
    elif age < 61:
        return "mid_adult"
    else:
        return "elderly"


labels_map = {"child": 0, "teenager": 1, "adult": 2, "mid_adult": 3, "elderly": 4}


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

    for i, file in enumerate(tqdm(files, total=len(files), desc=f"deepface_age | {ds}-{tech}")):
        if util._TEST_SINGLE and i > 0:
            break
        path_a = os.path.join(aligned_path, file)
        path_d = os.path.join(deid_path, file)
        if not os.path.exists(path_a):
            util.log(os.path.join(path_to_log, "deepface_age.txt"), f"({ds}) Missing: {path_a}")
            continue
        if not os.path.exists(path_d):
            util.log(os.path.join(path_to_log, "deepface_age.txt"), f"({tech}) Missing: {path_d}")
            continue

        pred_a = DeepFace.analyze(img_path=path_a, actions=["age"], detector_backend="skip")
        pred_d = DeepFace.analyze(img_path=path_d, actions=["age"], detector_backend="skip")

        age_a = round(pred_a[0].get("age", 0))
        age_d = round(pred_d[0].get("age", 0))
        bin_a = age_bin(age_a)
        bin_d = age_bin(age_d)

        is_match = 1 if bin_a == bin_d else 0
        metrics_df.add_score(file, is_match)
        metrics_df.add_column_value("aligned_age", age_a)
        metrics_df.add_column_value("deidentified_age", age_d)
        metrics_df.add_column_value("aligned_predictions", labels_map.get(bin_a, -1))
        metrics_df.add_column_value("deidentified_predictions", labels_map.get(bin_d, -1))

    metrics_df.save_to_csv(path_to_save)
    print(f"deepface_age saved into {path_to_save}")


if __name__ == "__main__":
    main()
