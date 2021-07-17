import os
import sys
from utils import download_RAVDESS
import logging

logging.basicConfig(filename="runtime.log", \
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
					datefmt='%d-%b-%y %H:%M:%S', \
					level=logging.DEBUG, filemode="a")

"""
Filename identifiers :
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
    Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
"""
# emotion_dict = dict()
# for i,j in zip(["01", "02", "03", "04", "05", "06", "07", "08"],["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]):
#     for k,l in zip(["01","02"],["strong", "normal"]):
#         emotion_dict[i+k] = "_".join([j,l])

# print(emotion_dict)
EMOTION_DICT={'0101': 'neutral_strong', '0102': 'neutral_normal',
                   '0201': 'calm_strong', '0202': 'calm_normal', '0301': 'happy_strong',
                   '0302': 'happy_normal', '0401': 'sad_strong', '0402': 'sad_normal',
                   '0501': 'angry_strong', '0502': 'angry_normal', '0601': 'fearful_strong',
                   '0602': 'fearful_normal', '0701': 'disgust_strong',
                   '0702': 'disgust_normal', '0801': 'surprised_strong', '0802': 'surprised_normal'}



if __name__=="__main__":
	RAVDESS_URL = "https://zenodo.org/record/1188976"
	logging.debug("\nCalling from train.py, to retrieve the required urls from {}.".format(RAVDESS_URL))
	urls = download_RAVDESS.RAVDESS_urls(RAVDESS_URL)
	logging.debug("\nRequired urls are retrieved.")
	logging.debug("\nDownloading the files from the urls.")
	download_RAVDESS.download_zip(urls)
	logging.debug("\nFinished Downloading the zipfiles. Moving to unzipping the zipfiles.")
	download_RAVDESS.unzip_audio_files()
	logging.debug("\nFinished unzipping.")
