"""DeepFace-based race classification evaluation."""
import os
from deepface import DeepFace
import utils as util
from tqdm import tqdm

# DeepFace outputs: White, Black, Asian, Indian, Arabic, Mixed
labels_map = {
    "white": 0, "black": 1, "asian": 2,
    "indian": 3, "arabic": 4, "mixed": 5,
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
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for i, file in enumerate(tqdm(files, total=len(files), desc=f"deepface_race | {ds}-{tech}")):
        if Path(file).suffix.lower() not in valid_extensions:
            continue
        if util._TEST_SINGLE and i > 0:
            break
        path_a = os.path.join(aligned_path, file)
        path_d = os.path.join(deid_path, file)
        if not os.path.exists(path_a):
            util.log(os.path.join(path_to_log, "deepface_race.txt"), f"({ds}) Missing: {path_a}")
            continue
        if not os.path.exists(path_d):
            util.log(os.path.join(path_to_log, "deepface_race.txt"), f"({tech}) Missing: {path_d}")
            continue

        pred_a = DeepFace.analyze(img_path=path_a, actions=["race"], detector_backend="skip")
        pred_d = DeepFace.analyze(img_path=path_d, actions=["race"], detector_backend="skip")

        ra = pred_a[0].get("dominant_race")
        rd = pred_d[0].get("dominant_race")
        is_match = 1 if ra == rd else 0
        metrics_df.add_score(file, is_match)
        metrics_df.add_column_value("aligned_predictions", labels_map.get(ra.lower(), -1))
        metrics_df.add_column_value("deidentified_predictions", labels_map.get(rd.lower(), -1))

    metrics_df.save_to_csv(path_to_save)
    print(f"deepface_race saved into {path_to_save}")


if __name__ == "__main__":
    main()
