import augme.video as amv
from pathlib import Path
import numpy as np
import csv
import os


AUDIO_DIR = "../audio/mp3"


def append_csv_row(csv_file, row):
    with open(csv_file, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(row)


def add_meta_header(csv_file):
    row = [
        "vid",
        "duration",
        "fps",
        "width",
        "height",
    ]
    append_csv_row(csv_file, row)


def add_query_meta_row(csv_file, query):
    row = [
        query["vid"],
        query["duration"],
        query["fps"],
        query["width"],
        query["height"],
    ]
    append_csv_row(csv_file, row)


def add_meta_row(csv_file, vid, info):
    row = [
        vid,
        info["duration"],
        info["r_frame_rate"],
        info["width"],
        info["height"],
    ]
    append_csv_row(csv_file, row)


def add_match_gt_header(csv_file):
    row = [
        "query_id",
        "ref_id",
        "query_start",
        "query_end",
        "ref_start",
        "ref_end",
    ]
    append_csv_row(csv_file, row)


def add_stack_meta_header(csv_file):
    row = [
        "vid",
        "duration",
        "fps",
        "width",
        "height",
    ]
    append_csv_row(csv_file, row)


def add_stack_gt_header(csv_file):
    row = [
        "vid",
        "stack",
    ]
    append_csv_row(csv_file, row)


def add_stack_gt_row(csv_file, vid, stack):
    row = [
        vid,
        stack,
    ]
    append_csv_row(csv_file, row)


def add_mp3_header(csv_file):
    row = [
        "aid",
        "duration",
        "original_path",
    ]
    append_csv_row(csv_file, row)


def add_mp3_row(csv_file, aid, info, path):
    row = [
        aid,
        info["duration"],
        path.as_posix(),
    ]
    append_csv_row(csv_file, row)


def get_mp3_paths(audio_path):
    mp3_paths = set() # main video set
    mp3_dir_list = list(Path(audio_path).glob("*"))
    for mp3_dir in mp3_dir_list:
        mp3_list = list(mp3_dir.glob("*.mp3"))
        for mp3 in mp3_list:
            mp3_paths.add(mp3)
    return list(mp3_paths)
    

def get_stack(meta_list):
    match meta_list[0]["name"]:
        case 'hstack':
            return 'hstack'
        case 'vstack':
            return 'vstack'
        case 'fstack':
            return 'fstack'
        case _:
            return 'nstack'


def get_subset_id(subset):
    match subset:
        case 'train':
            return '1'
        case 'val':
            return '2'
        case 'test':
            return '3'
        case _:
            return '0'


def select_random_audio(audio_path, rng=np.random.default_rng()):
    audio_path_list = list(Path(audio_path).glob("*.mp3"))
    audio = rng.choice(audio_path_list)
    
    return str(audio)


def get_font_path(rng=np.random.default_rng()):
    # ffmpeg only accepts posix-style relative directory of fontfile
    working_directory = os.path.dirname(__file__)
    font_path = amv.helpers.select_random_font(rng=rng)
    font_path = os.path.relpath(font_path, working_directory)
    font_path = Path(font_path).as_posix()

    return str(font_path)


def add_mp4_header(csv_file):
    row = [
        "original_id",
        "path",
    ]
    append_csv_row(csv_file, row)


def get_mp4_paths(video_path):
    mp4_paths = set() # main video set
    first_dirs = list(Path(video_path).glob("*"))
    for first_dir in first_dirs:
        second_dirs = first_dir.glob("*")
        for second_dir in second_dirs:
            video_paths = second_dir.glob("*")
            for video_path in video_paths:
                mp4_paths.add(video_path)
    return list(mp4_paths)


def add_mp4_row(csv_file, path):
    row = [
        path.stem,
        path.as_posix(),
    ]
    append_csv_row(csv_file, row)


def add_vid_header(csv_file):
    row = [
        "vid",
        "original_id",
    ]
    append_csv_row(csv_file, row)


def add_vid_row(csv_file, vid, path):
    row = [
        vid,
        Path(path).stem,
    ]
    append_csv_row(csv_file, row)


def select_augment_method(
        second_video_path,
        third_video_path=None,
        fourth_video_path=None,
        audio_dir=AUDIO_DIR,
        start=None,
        end=None,
        rng=np.random.default_rng(),
    ):
    noise_level = rng.integers(20000, 1000000)
    audio_path = select_random_audio(audio_dir, rng=rng)
    blend_opacity = rng.uniform(0.2, 0.8)
    blur_sigma = rng.uniform(1, 10)
    brightness_level = rng.uniform(-0.6, 0.6)
    aspect_ratio = rng.uniform(9/16, 16/9)
    speed_factor = rng.uniform(0.3, 3.0)
    saturation_factor = rng.uniform(0.0, 3.0)
    pad_color = (rng.integers(0, 255), rng.integers(0, 255), rng.integers(0, 255))
    contrast_level = rng.uniform(-2.0, 2.0)
    crop_left = rng.uniform(0, 0.3)
    crop_top = rng.uniform(0, 0.3)
    crop_right = rng.uniform(0.7, 1.0)
    crop_bottom = rng.uniform(0.7, 1.0)
    encoding_quality = rng.integers(17, 51)
    fps = rng.integers(1, 30)
    num_loops = rng.integers(1, 3)
    overlay_size = rng.uniform(0.3, 0.8)
    overlay_x = rng.uniform(0, 1 - overlay_size)
    overlay_y = rng.uniform(0, 1 - overlay_size)
    emoji_path = amv.helpers.select_random_emoji(rng=rng)
    emoji_x = rng.uniform(0.1, 0.6)
    emoji_y = rng.uniform(0.1, 0.6)
    emoji_opacity = rng.uniform(0.5, 1.0)
    emoji_size = rng.uniform(0.2, 0.6)
    text_lines = rng.integers(1, 10)
    text_fontsize = rng.uniform(0.05, 1/text_lines)
    text_opacity = rng.uniform(0.5, 1.0)
    font_path = get_font_path(rng=rng)
    pad_width = rng.uniform(0, 0.25)
    pad_height = rng.uniform(0, 0.25)
    pixelization_factor = rng.uniform(0.1, 0.5)
    frames_cache = rng.integers(2, 16)
    resize_width = rng.integers(320, 1920)
    resize_height = rng.integers(320, 1920)
    rotate_degrees = rng.choice([i for i in range(-30, 30, 5)])
    rotate90 = rng.choice([i for i in range(-270, 270, 90)])
    scale_factor = rng.uniform(0.2, 0.8)
    transpose_direction = rng.choice([0, 1, 2, 3])
    stack_pad = rng.choice([True, False])
    stack_preserve_aspect_ratio = rng.choice([True, False])
    stack_2grid = rng.choice([0, 1])
    stack_4grid = rng.choice([0, 1, 2, 3])

    # note the order: modify time are first two, add stack are last four
    augment_methods = [
        amv.ChangeVideoSpeed(factor=speed_factor),
        amv.Loop(num_loops=num_loops),
        amv.AddNoise(level=noise_level, add_audio_noise=True),
        amv.AudioSwap(audio_path=audio_path, audio_offset=0),
        amv.BlendVideos(overlay_path=second_video_path, opacity=blend_opacity, merge_audio=False),
        amv.Blur(sigma=blur_sigma),
        amv.Brightness(level=brightness_level),
        amv.ChangeAspectRatio(ratio=aspect_ratio),
        amv.ColorJitter(saturation_factor=saturation_factor),
        amv.Contrast(level=contrast_level),
        amv.Crop(left=crop_left, top=crop_top, right=crop_right, bottom=crop_bottom),
        amv.EncodingQuality(quality=encoding_quality),
        amv.FPS(fps=fps),
        amv.Gradient(),
        amv.Grayscale(),
        amv.HFlip(),
        amv.Overlay(overlay_path=second_video_path, overlay_size=overlay_size, x_factor=overlay_x,
            y_factor=overlay_y, merge_audio=False),
        amv.OverlayEmoji(emoji_path=emoji_path, x_factor=emoji_x, y_factor=emoji_y,
            opacity=emoji_opacity, emoji_size=emoji_size),
        amv.OverlayText(font=font_path, fontsize=text_fontsize, num_lines=text_lines,
            opacity=text_opacity),
        amv.Pad(w_factor=pad_width, h_factor=pad_height, color=pad_color),
        amv.Pixelization(ratio=pixelization_factor),
        amv.RandomFrames(num_frames=frames_cache),
        amv.RemoveAudio(),
        amv.Resize(width=resize_width, height=resize_height),
        amv.Rotate(degrees=rotate_degrees),
        amv.Scale(factor=scale_factor),
        amv.VFlip(),
        amv.Transpose(direction=transpose_direction),
        amv.VStack(
            second_video_path=second_video_path,
            merge_audio=True,
            target_grid=stack_2grid,
            preserve_aspect_ratio=stack_preserve_aspect_ratio,
            pad_second_video=stack_pad,
            pad_color=pad_color
        ),
        amv.HStack(
            second_video_path=second_video_path,
            merge_audio=True,
            target_grid=stack_2grid,
            preserve_aspect_ratio=stack_preserve_aspect_ratio,
            pad_second_video=stack_pad,
            pad_color=pad_color
        ),
        amv.FStack(
            second_video_path=second_video_path,
            third_video_path=third_video_path,
            fourth_video_path=fourth_video_path,
            merge_audio=True,
            target_grid=stack_4grid,
            preserve_aspect_ratio=stack_preserve_aspect_ratio,
            pad_video=stack_pad,
            pad_color=pad_color
        ),
    ]

    return rng.choice(augment_methods[start:end])
