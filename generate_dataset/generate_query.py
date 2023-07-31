import augme.video as amv
import os 
from pathlib import Path
import argparse
import time
import tempfile
import json
import shutil
import pandas as pd
import numpy as np
import utils


def main(args):
    """Create query for similar video detection and localization tasks."""
    data_path = args.data_path
    subset = args.subset
    subset_path = f"{data_path}/{subset}"

    csv_reference_meta = f"{subset_path}/{subset}_reference_metadata.csv"
    csv_query_raw_meta = f"{subset_path}/{subset}_query_raw_metadata.csv"
    csv_distractor_meta = f"{subset_path}/{subset}_distractor_metadata.csv"

    csv_match_gt = f"{subset_path}/{subset}_matching_ground_truth.csv"
    csv_query_meta = f"{subset_path}/{subset}_query_metadata.csv"

    # create csv files if not already exist and add the header
    if not Path(csv_match_gt).is_file():
        os.makedirs(f"{subset_path}/query", exist_ok=True)
        os.makedirs(f"{subset_path}/query_metadata", exist_ok=True)

        utils.add_meta_header(csv_query_meta)
        utils.add_match_gt_header(csv_match_gt)

    df_reference_meta = pd.read_csv(csv_reference_meta)
    df_query_raw_meta = pd.read_csv(csv_query_raw_meta)
    df_distractor_meta = pd.read_csv(csv_distractor_meta)

    for index, query in df_query_raw_meta.iterrows():
        query_path = f"{subset_path}/query_raw/{query['vid']}.mp4"
        query_output_path = f"{subset_path}/query/{query['vid']}.mp4"
        out_meta_path = f"{subset_path}/query_metadata/{query['vid']}.json"
        # choose whether to insert a reference segment
        extract_segment = rng.choice([True, True, False])

        if (not extract_segment) or query['duration'] < 5:
            shutil.copyfile(query_path, query_output_path)
            utils.add_query_meta_row(csv_query_meta, query)
            continue

        if extract_segment:
            # choose how many segments to insert
            num_segments = rng.choice([1,2,3])

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_query = os.path.join(tmpdir, f"{os.urandom(24).hex()}.mp4")
                segment_index = 0
                video_paths = []
                query_offsets = [0]
                segment_durations = []
                meta_list = []

                while segment_index < num_segments:
                    reference = df_reference_meta.sample(n=1, replace=False, random_state=rng)
                    if reference.iloc[0]['duration'] > 5:
                        # choose reference segment to insert
                        segment_duration = rng.uniform(2, min(reference.iloc[0]['duration'] - 1, 60))
                        ref_start = rng.uniform(0, reference.iloc[0]['duration'] - segment_duration)
                        ref_end = ref_start + segment_duration

                        reference_path = f"{subset_path}/reference/{reference.iloc[0]['vid']}.mp4"
                        # choose the time point where to insert the reference segment into query
                        query_offset = rng.uniform(
                            segment_index * query["duration"] / num_segments + 0.7,
                            (segment_index + 1) * query["duration"] / num_segments - 0.7
                        )

                        try:
                            query_before_path = os.path.join(tmpdir, f"{os.urandom(24).hex()}.mp4")
                            amv.trim(query_path, query_before_path, start=query_offsets[-1], end=query_offset)

                            segment_path = os.path.join(tmpdir, f"{os.urandom(24).hex()}.mp4")
                            amv.trim(reference_path, segment_path, start=ref_start, end=ref_end)
                            
                            distractors = df_distractor_meta.sample(n=1, replace=False, random_state=rng)
                            distractor_path = f"{subset_path}/distractor/{distractors.iloc[0]['vid']}.mp4"

                            # augment each segment separately
                            segment_augment_path = os.path.join(tmpdir, f"{os.urandom(24).hex()}.mp4")
                            augment_method = utils.select_augment_method(distractor_path, end=-3, rng=rng)
                            augment_method(segment_path, segment_augment_path, metadata=meta_list)

                            # update segment duration
                            if meta_list[-1]["name"] == "loop":
                                segment_duration *= (1 + meta_list[-1]["num_loops"])
                            if meta_list[-1]["name"] == "change_video_speed":
                                segment_duration /= meta_list[-1]["factor"]

                            match_gt_row = [
                                query['vid'],
                                reference.iloc[0]['vid'],
                                query_offset + sum(segment_durations),
                                query_offset + sum(segment_durations) + segment_duration,
                                    ref_start,
                                    ref_end,
                                ]
                        except:
                            # in case of error try another reference video
                            continue
                        video_paths.append(query_before_path)
                        video_paths.append(segment_augment_path)

                        utils.append_csv_row(csv_match_gt, match_gt_row)
                        query_offsets.append(query_offset)
                        segment_durations.append(segment_duration)
                        segment_index += 1
                    else:
                        # if reference video <= 5, choose another longer reference video
                        continue

                try:
                    # add the rightmost cut of query
                    if query_offsets[-1] < query["duration"]:
                        query_after_path = os.path.join(tmpdir, f"{os.urandom(24).hex()}.mp4")
                        amv.trim(query_path, query_after_path, start=query_offsets[-1])
                        video_paths.append(query_after_path)

                    # concatenate query parts and reference segments
                    query_concat_path = os.path.join(tmpdir, f"{os.urandom(24).hex()}.mp4")
                    amv.concat(video_paths, query_concat_path)

                    # distractors are fillers to query (stack, overlay, etc.)
                    distractors = df_distractor_meta.sample(n=3, replace=False, random_state=rng)
                    distractor_path0 = f"{subset_path}/distractor/{distractors.iloc[0]['vid']}.mp4"
                    distractor_path1 = f"{subset_path}/distractor/{distractors.iloc[1]['vid']}.mp4"
                    distractor_path2 = f"{subset_path}/distractor/{distractors.iloc[2]['vid']}.mp4"

                    augment_method = utils.select_augment_method(
                        distractor_path0,
                        distractor_path1,
                        distractor_path2,
                        start=2,
                        rng=rng,
                    )
                    augment_method(query_concat_path, query_output_path, metadata=meta_list)
                    query_output_info = amv.helpers.get_video_info(query_output_path)
                except:
                    # in case of error undo csv row updates and move to the next query video
                    df_match_gt = pd.read_csv(csv_match_gt)
                    df_match_gt = df_match_gt.iloc[:-num_segments]
                    df_match_gt.to_csv(csv_match_gt, index=False)

                    shutil.copyfile(query_path, query_output_path)
                    utils.add_query_meta_row(csv_query_meta, query)
                    continue

                utils.add_meta_row(csv_query_meta, query["vid"], query_output_info)

                with open(out_meta_path, "w") as f:
                    json.dump(meta_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../vsl_dataset")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    main(args)
    print("### COMPLETED IN %.3f SECONDS ###" % (time.time() - start_time))
