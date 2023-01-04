# -*- coding: utf-8 -*-
"""
Prep_and_Publish_Audio_Dataset_from_S3.ipynb

"""


"""Set up S3 session"""

from huggingface_hub import notebook_login

notebook_login()

import datasets

s3 = datasets.filesystems.S3FileSystem(key='AKIARYVVJ52TE25M3YFZ', secret='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')
s3_root_path="radio-dataset"


# Create dataset

import torchaudio
import librosa

from datasets import load_dataset, load_metric, Audio, DatasetDict,Dataset
sampling_rate = 16000
dataset_name = "mutisya/swahili_radio_2021_v0.1_bbc"

datasetPath="s3://radio-dataset/swahili/"

allFiles = []

allFiles.extend(s3.ls(datasetPath+"bbc"))
#allFiles.extend(s3.ls(datasetPath+"dw"))
#allFiles.extend(s3.ls(datasetPath+"UN"))
#allFiles.extend(s3.ls(datasetPath+"VoA"))

len(allFiles)

"""Read files from S3 in streaming fashion and write them into the dataset"""

import tempfile
import warnings

def read_audio_s3_items_to_dataset(allFiles, s3):
    totalSeconds = 0
    def gen():
        nonlocal totalSeconds
        for item in allFiles:
            if item.endswith(".wav"):
                with s3.open(item, 'rb') as f:
                    arr, sr = librosa.load(f, sr=sampling_rate)
                    totalSeconds += librosa.get_duration(y=arr, sr=sr)
                    yield {"audio": arr}
            if item.endswith(".mp3"):
                tmp = tempfile.NamedTemporaryFile(suffix = '.mp3')
                with open(tmp.name, 'wb') as f:
                    s3.download(item, tmp.name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arr, sr = librosa.load(tmp.name, sr=sampling_rate)
                        totalSeconds += librosa.get_duration(y=arr, sr=sr)
                        yield {"audio": arr}

    dataset = Dataset.from_generator(gen)
    return dataset,totalSeconds



"""Save dataset"""

# dataset.push_to_hub(dataset_name, private=True)

from huggingface_hub import HfApi

def upload_file_to_dataset(allFiles, s3):
# file length info to file
length_tracking_file = "DatasetLength.md"

with open(length_tracking_file, "a") as file_object:
            file_object.write("Total Seconds: "+ totalSeconds)
            file_object.write("\n")
            file_object.write("Total Minutes: "+ totalSeconds/60)
            file_object.write("\n")
            file_object.write("Total Hours: "+ totalSeconds/3600)
            file_object.write("\n")

api = HfApi()
api.upload_file(
    path_or_fileobj=length_tracking_file,
    path_in_repo=length_tracking_file,
    repo_id=dataset_name,
    repo_type="dataset",
)

# we could use this approach to modify the ReadMe file for the dataset