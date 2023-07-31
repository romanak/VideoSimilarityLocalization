import augme.video as amv
import os 
from pathlib import Path
import argparse
import time
import shutil
import csv
import pandas as pd
import numpy as np
import utils
from math import ceil


def create_subset(
        num_reference,
        num_query,
        num_distractor,
        subset,
        args,
        exclude_list=[],
        rng=np.random.default_rng(),
    ):
    """Randomly choose videos from Multimedia Commons dataset."""
    subset_path = f"{args.output_path}/{subset}"
    subset_id = utils.get_subset_id(subset)

    csv_mp4_paths = f"{args.output_path}/mp4_paths.csv"
    csv_reference_meta = f"{subset_path}/{subset}_reference_metadata.csv"
    csv_query_raw_meta = f"{subset_path}/{subset}_query_raw_metadata.csv"
    csv_distractor_meta = f"{subset_path}/{subset}_distractor_metadata.csv"
    csv_vids = f"{subset_path}/{subset}_vids.csv"

    if not Path(csv_mp4_paths).is_file():
        os.makedirs(f"{args.output_path}", exist_ok=True)
        utils.add_mp4_header(csv_mp4_paths)
        mp4_paths = utils.get_mp4_paths(args.data_path)
        for path in mp4_paths:
            utils.add_mp4_row(csv_mp4_paths, path)

    if not Path(csv_reference_meta).is_file():
        os.makedirs(f"{subset_path}/reference", exist_ok=True)
        os.makedirs(f"{subset_path}/query_raw", exist_ok=True)
        os.makedirs(f"{subset_path}/distractor", exist_ok=True)

        utils.add_meta_header(csv_reference_meta)
        utils.add_meta_header(csv_query_raw_meta)
        utils.add_meta_header(csv_distractor_meta)
        utils.add_vid_header(csv_vids)

    df_mp4_paths = pd.read_csv(csv_mp4_paths)
    df_mp4_paths = df_mp4_paths[~df_mp4_paths['original_id'].isin(exclude_list)]

    # choose subsets
    num_all = num_reference + num_query + num_distractor
    df_dataset = df_mp4_paths.sample(n=num_all, replace=False, random_state=rng)

    df_reference = df_dataset.iloc[:num_reference]
    df_query = df_dataset.iloc[num_reference:num_reference+num_query]
    df_distractor = df_dataset.iloc[num_reference+num_query:]
    exclude_list = df_dataset['original_id'].tolist()

    i = 1
    for index, row in df_reference.iterrows():
        reference = row['path']
        reference_info = amv.helpers.get_video_info(reference)
        if reference_info:
            vid = f"R{subset_id}{str(i).zfill(5)}"
            shutil.copyfile(reference, f"{subset_path}/reference/{vid}.mp4")
            utils.add_meta_row(csv_reference_meta, vid, reference_info)
            utils.add_vid_row(csv_vids, vid, reference)
            i += 1

    i = 1
    for index, row in df_query.iterrows():
        query = row['path']
        vid = f"Q{subset_id}{str(i).zfill(5)}"
        query_info = amv.helpers.get_video_info(query)
        if query_info:
            shutil.copyfile(query, f"{subset_path}/query_raw/{vid}.mp4")
            utils.add_meta_row(csv_query_raw_meta, vid, query_info)
            utils.add_vid_row(csv_vids, vid, query)
            i += 1

    i = 1
    for index, row in df_distractor.iterrows():
        distractor = row['path']
        vid = f"D{subset_id}{str(i).zfill(5)}"
        distractor_info = amv.helpers.get_video_info(distractor)
        if distractor_info:
            shutil.copyfile(distractor, f"{subset_path}/distractor/{vid}.mp4")
            utils.add_meta_row(csv_distractor_meta, vid, distractor_info)
            utils.add_vid_row(csv_vids, vid, distractor)
            i += 1

    return exclude_list


def main(args):
    rng = np.random.default_rng(args.seed)
    subset_size = int(args.subset_size)

    train_exclude_list = create_subset(
        num_reference=max(640, subset_size),
        num_query=max(160, ceil(subset_size/4)),
        num_distractor=max(40, ceil(subset_size/16)),
        subset="train",
        args=args,
        exclude_list=[],
        rng=rng,
    )

    val_exclude_list = create_subset(
        num_reference=max(64, ceil(subset_size/10)),
        num_query=max(16, ceil(subset_size/40)),
        num_distractor=max(4, ceil(subset_size/160)),
        subset="val",
        args=args,
        exclude_list=train_exclude_list,
        rng=rng,
    )

    test_exclude_list = create_subset(
        num_reference=max(320, ceil(subset_size/2)),
        num_query=max(80, ceil(subset_size/8)),
        num_distractor=max(20, ceil(subset_size/32)),
        subset="test",
        args=args,
        exclude_list=val_exclude_list + train_exclude_list,
        rng=rng,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/path/to/multimedia-commons/mp4")
    parser.add_argument("--output_path", type=str, default="../vsl_dataset")
    parser.add_argument("--subset_size", type=int, default="400")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("### COMPLETED IN %.3f SECONDS ###" % (time.time() - start_time))
