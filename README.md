# Multimodal Video Similarity Detection and Localization Using Audio-Visual Features

## Create Video Similarity Localization dataset

Download (8.5Gb) the [Video Similarity Localization small dataset](https://tku365-my.sharepoint.com/:f:/g/personal/610785015_o365_tku_edu_tw/EtpZAWxJHphKonletxX0SrwBCqRR2O2jVFHKVPw5l-msVw?e=Vjxs5C) consisting of a subset of 1,344 videos (--subset_size 640) for experiments. It has 329 temporally annotated matched segments.

Download (552Gb) [Video Similarity Localization dataset](https://tku365-my.sharepoint.com/:f:/g/personal/610785015_o365_tku_edu_tw/EuMiWi1h77hCrPPkjoyn8goBPHDPRzfUKs5nt5rrXyKjvw?e=W1Z7if) consisting of a subset of 84,795 videos (--subset_size 40000). It has 20,445 temporally annotated matched segments.

To create a new instance of the Video Similarity Localization dataset, go through the following steps.

> Note: sometimes FFmpeg may occasionally stall on a specific video (reasons unknown). In such a case, [AugMe](https://github.com/romanak/AugMe) will wait 1 hour timeout before breaking out of the subprocess; then try/catch will move on to the next video. However, before running the code you may want to remove try/catch clauses to ensure everything runs smoothly (otherwise there may be an infinite loop).

### Download Multimedia Commons dataset

Download the [Multimedia Commons](http://multimediacommons.org/) video dataset (~2.7Tb, ~787K videos) by using AWS CLI. Follow instructions on their website.

### Install required packages

Create a virtual environment and install the required packages (example for Windows), then go to `generate_dataset` directory:

```plaintext
cd VideoSimilarityLocalization
python -m venv .env
.env/Scripts/activate
pip install -r generate_dataset/requirements.txt
cd generate_dataset
```

## Prepare audio data

Create a directory `../audio/mp3` and put a few mp3 audio files in it. This is necessary for `Generate query data` step for `audio_swap` augmentation to randomly choose audio from this directory. Optionally, you can exclude this augmentation from `augment_methods` list in `generate_dataset/utils.py`.

I used [fma_small.zip](https://os.unil.cloud.switch.ch/fma/fma_small.zip) of [Free Music dataset](https://github.com/mdeff/fma). If you want to use the same dataset, put all mp3 files in one directory by running the following command, specifying the path to the Free Music dataset:

```plaintext
python generate_audio_paths.py --audio_path /path/to/free_music_dataset --output_path ../audio
```

### Generate raw VSL dataset

In order to create a raw (unprocessed) Video Similarity Localization dataset, choose a subset of Multimedia Commons video dataset and split the videos into training, validation, and testing subsets. To complete this step, run the following command (indicate the correct path to the Multimedia Commons dataset and adjust the subset size if necessary):

```plaintext
python generate_raw_dataset.py --data_path /path/to/multimedia-commons/mp4 --output_path ../vsl_dataset --subset_size 640
```

### Generate query data

Query augmentations include pasting random segments of reference videos into a raw query video and video augmentations of each reference video segment, as well as augmentation of the concatenated query video. For this step, run the following command, specifying subsets to add query:

```plaintext
python generate_query.py --data_path ../vsl_dataset --subset train
```

### Generate stack dataset

Stack dataset is used to train a classification model to predict the stack class of the video: no stack, vertical stack, horizontal stack, or grid of four stack. To create the dataset, run the following command, adjusting the subset and the sample range if necessary. Sample range determines how many stacked videos will be created as well as their id.

```plaintext
python generate_stack_dataset.py --data_path ../vsl_dataset --subset train --sample_range 1 161
```

## Acknowledgments and Licenses

Please cite my work if you use my code or data.

```
@mastersthesis{akchurin2023multimodal,
  author  = "Roman Akchurin",
  title   = "Multimodal Video Similarity Detection and Localization Using Audio-Visual Features",
  school  = "Tamkang University",
  year    = "2023",
  address = "Taiwan",
}
```

* The code in this repository is released under the [MIT license](LICENSE).

* The dataset is meant for research purposes.
