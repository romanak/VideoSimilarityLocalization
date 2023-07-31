import augme.video as amv
import os
from pathlib import Path
import argparse
import time
import tempfile
import json
import pandas as pd
import numpy as np
import utils
import shutil


def main(args):
    """Generate a dataset for stack classification task."""
    data_path = args.data_path
    subset = args.subset
    subset_path = f"{data_path}/{subset}"
    subset_id = utils.get_subset_id(subset)

    csv_stack_meta = f"{subset_path}/{subset}_stack_metadata.csv"
    csv_stack_gt = f"{subset_path}/{subset}_stack_ground_truth.csv"

    csv_query_raw_meta = f"{subset_path}/{subset}_query_raw_metadata.csv"
    csv_distractor_meta = f"{subset_path}/{subset}_distractor_metadata.csv"
    csv_reference_meta = f"{subset_path}/{subset}_reference_metadata.csv"

    # create csv files if not already exist and add the header
    if not Path(csv_stack_gt).is_file():
        os.makedirs(f"{subset_path}/stack", exist_ok=True)
        os.makedirs(f"{subset_path}/stack_metadata", exist_ok=True)

        utils.add_stack_meta_header(csv_stack_meta)
        utils.add_stack_gt_header(csv_stack_gt)

    df_query_raw_meta = pd.read_csv(csv_query_raw_meta)
    df_distractor_meta = pd.read_csv(csv_distractor_meta)
    df_reference_meta = pd.read_csv(csv_reference_meta)

    start_index = args.sample_range[0]
    stop_index = args.sample_range[1]
    gen_index = start_index

    while gen_index < stop_index:
        query = df_query_raw_meta.sample(n=1, replace=False, random_state=rng)
        query_path = f"{subset_path}/query_raw/{query.iloc[0]['vid']}.mp4"
        if query.iloc[0]['duration'] < 5:
            continue

        # for better generalization, use reference videos for distractors
        references = df_reference_meta.sample(n=4, replace=False, random_state=rng)
        distractor_path0 = f"{subset_path}/reference/{references.iloc[0]['vid']}.mp4"
        distractor_path1 = f"{subset_path}/reference/{references.iloc[1]['vid']}.mp4"
        distractor_path2 = f"{subset_path}/reference/{references.iloc[2]['vid']}.mp4"
        distractor_path3 = f"{subset_path}/reference/{references.iloc[3]['vid']}.mp4"

        meta_list = []
        output_path = f"{subset_path}/stack/S{subset_id}{str(gen_index).zfill(5)}.mp4"
        out_meta_path = f"{subset_path}/stack_metadata/S{subset_id}{str(gen_index).zfill(5)}.json"

        apply_stack = rng.choice([False, True, True, True])

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, f"{os.urandom(24).hex()}.mp4")

            try:
                if apply_stack:
                    augment_method = utils.select_augment_method(
                        distractor_path0,
                        distractor_path1,
                        distractor_path2,
                        start=-3,
                        rng=rng,
                    )
                else:
                    augment_method = utils.select_augment_method(distractor_path0, start=2, end=-3, rng=rng)
                
                augment_method(query_path, temp_path, metadata=meta_list)
                # do not include transpose!
                augment_method = utils.select_augment_method(distractor_path3, start=2, end=-4, rng=rng)
                augment_method(temp_path, output_path, metadata=meta_list)

                output_info = amv.helpers.get_video_info(output_path)
                stack = utils.get_stack(meta_list)
            except:
                continue

            utils.add_meta_row(csv_stack_meta, f"S{subset_id}{str(gen_index).zfill(5)}", output_info)
            utils.add_stack_gt_row(csv_stack_gt, f"S{subset_id}{str(gen_index).zfill(5)}", stack)
            
            with open(out_meta_path, "w") as f:
                json.dump(meta_list, f)

            # extract individual frames
            stack_frames_dir = f"{subset_path}/stack_frames/{stack}"
            amv.helpers.extract_frames(output_path, stack_frames_dir, fps=1)
            
            gen_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../vsl_dataset")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_range", nargs="*", type=int, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    main(args)
    print("### COMPLETED IN %.3f SECONDS ###" % (time.time() - start_time))
