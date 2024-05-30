import csv
import os
import itertools
import random

LABELS_FOLDER = "root_dir/datasets/labels"
PAIRS_FOLDER = "root_dir/datasets/pairs"
ALIGNED_FOLDER ='root_dir/datasets/aligned'


def count_lines(file_path):
    count = 0
    with open(file_path, "r") as file:
        for line in file:
            count += 1
    return count


def main(selected_datasets_names: list):
    # Set a random seed for reproducibility
    random.seed(42)

    for dataset_name in selected_datasets_names:
        genuine_pairs_savepath = os.path.join(
            PAIRS_FOLDER, dataset_name + "_genuine_pairs.txt"
        )
        impostor_pairs_savepath = os.path.join(
            PAIRS_FOLDER, dataset_name + "_impostor_pairs.txt"
        )
        # aligned_path = os.path.join(ALIGNED_FOLDER,dataset_name)
        # list_aligned_img = os.listdir(aligned_path)
        # Some img's extension have been changed so we do not take it in account here
        # for i in range(len(list_aligned_img)):
        #     list_aligned_img[i] = list_aligned_img[i].split('.')[0]

        # print("\n aligned images : \n",list_aligned_img)

        # Load identities and their corresponding images
        identity_clusters = {}
        with open(
            os.path.join(LABELS_FOLDER, dataset_name + "_labels.csv"), "r"
        ) as labels_file:
            csv_reader = csv.DictReader(labels_file)
            first_row = next(csv_reader, None)
            if first_row["Identity"]=="":
                print(
                    f"no id label available for {dataset_name} dataset, we can't generate pairs"
                )
                continue
            
            # Continue processing if identity labels are available
            for line in csv_reader:
                # print("img_name: ",line["Name"].split('.')[0] )
                # if line["Name"].split('.')[0] in list_aligned_img: # we use only aligned img to generate pairs
                identity = line["Identity"]
                img_name = line["Name"]

                if identity in identity_clusters:
                    identity_clusters[identity].append(img_name)
                else:
                    identity_clusters[identity] = [img_name]
            # print("\nidentity clusters:\n", identity_clusters)

        # if not os.path.exists(genuine_pairs_savepath):
        # Generate genuine pairs
        genuine_pairs = []
        for identity, images in identity_clusters.items():
            if len(images) >= 2:
                genuine_pairs.extend(list(itertools.combinations(images, 2)))
        number_of_pairs = len(genuine_pairs)
        # Save genuine pairs to a text file
        with open(genuine_pairs_savepath, "w+") as genuine_pairs_file:
            for pair in genuine_pairs:
                id1 = [
                    id
                    for id, images in identity_clusters.items()
                    if pair[0] in images
                ][0]
                id2 = [
                    id
                    for id, images in identity_clusters.items()
                    if pair[1] in images
                ][0]
                genuine_pairs_file.write(f"{id1} {pair[0]} {id2} {pair[1]}\n")
        print(f"Genuine pairs generated for {dataset_name} dataset.")
        # else:
        #     print(f"Genuine pairs for {dataset_name} dataset already exist.")
        #     number_of_pairs = count_lines(genuine_pairs_savepath)

        # Generate impostor pairs
        if len(identity_clusters) < 2:
            print(
                f"Not enough identities to generate impostor pairs for {dataset_name} dataset. Skipping impostor pairs generation."
            )
        else:
            impostor_pairs = []
            for _ in range(number_of_pairs):
                identity_a, identity_b = random.sample(
                    list(identity_clusters.keys()), 2
                )
                impostor_pairs.append(
                    (
                        random.choice(identity_clusters[identity_a]),
                        random.choice(identity_clusters[identity_b]),
                    )
                )

            # Save impostor pairs to a text file
            with open(impostor_pairs_savepath, "w+") as impostor_pairs_file:
                for pair in impostor_pairs:
                    id1 = [
                        id
                        for id, images in identity_clusters.items()
                        if pair[0] in images
                    ][0]
                    id2 = [
                        id
                        for id, images in identity_clusters.items()
                        if pair[1] in images
                    ][0]
                    impostor_pairs_file.write(f"{id1} {pair[0]} {id2} {pair[1]}\n")

            print(f"Impostor pairs generated for {dataset_name} dataset.")


# if __name__ == "__main__":
#     main(["fri"])
