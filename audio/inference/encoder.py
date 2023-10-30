import opensmile
import numpy as np
import torch.nn as nn
import torch
import os
from moviepy.editor import VideoFileClip

def opensmile_feature(audio_file_path, feature_type):
    if feature_type == "emobase":
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    elif feature_type == "ComParE":
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return np.array(smile.process_file(audio_file_path))


def get_emb_from_audio(audio_array):
    # Z-score standardization
    mean = np.mean(audio_array)
    std_dev = np.std(audio_array)
    standardized_a= (audio_array - mean) / std_dev

    # Dimensionality reduction to (300,)
    fc_layer = nn.Linear(6373, 300)
    a_ = torch.tensor(standardized_a).float()
    output_tensor = fc_layer(a_).detach().numpy().reshape(-1,)
    
    return output_tensor



def get_audio_from_mp4(video_filename):
    filename, _ = os.path.splitext(video_filename)
    clip = VideoFileClip(video_filename)
    clip.audio.write_audiofile(f"{filename}.wav", verbose=False, logger=None)
    return f"{filename}.wav"
    
def get_audio_emb_from_one_mp4(video_filename):
    audio_filename = get_audio_from_mp4(video_filename)
    audio_array = opensmile_feature(audio_filename, 'ComParE')
    return get_emb_from_audio(audio_array)

def make_embs_for_dialogue(dialogue_dir):
    video_filenames = [f for f in os.listdir(dialogue_dir) if os.path.isfile(os.path.join(dialogue_dir, f))]
    video_filenames.sort()
    video_filenames = [os.path.join(dialogue_dir, f) for f in video_filenames]
    video_filenames = [f for f in video_filenames if f.endswith('.mp4')]
    
    # store all the n (300,) arrays as one (n, 300) matrix where n is the number of files in the dialogue_dir
    audio_embs = np.array([get_audio_emb_from_one_mp4(video_filename) for video_filename in video_filenames])
    return audio_embs