import os 
from pathlib import Path
import argparse
import time
import shutil
import csv
import augme.video as amv
import utils


def main(args):
    csv_mp3_paths = f"{args.output_path}/mp3_paths.csv"
    # create csv files if not already exist and add the header
    if not Path(csv_mp3_paths).is_file():
        os.makedirs(f"{args.output_path}/mp3", exist_ok=True)
        utils.add_mp3_header(csv_mp3_paths)
        mp3_paths = utils.get_mp3_paths(args.audio_path)
        print(len(mp3_paths))
        for i in range(len(mp3_paths)):
            aid = f"A1{str(i).zfill(5)}"
            mp3_path = mp3_paths[i]
            audio_info = amv.helpers.get_audio_info(mp3_path)
            shutil.copyfile(mp3_path, f"{args.output_path}/mp3/{aid}.mp3")
            utils.add_mp3_row(csv_mp3_paths, aid, audio_info, mp3_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="/path/to/free_music_dataset")
    parser.add_argument("--output_path", type=str, default="../audio")
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("### COMPLETED IN %.3f SECONDS ###" % (time.time() - start_time))