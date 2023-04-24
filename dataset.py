import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
import matplotlib.pyplot as plt

def dataset(path):
    EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust',
                0: 'surprise'}
    DATA_PATH = path
    SAMPLE_RATE = 48000

    data = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender', 'Path'])
    for dirname, _, filenames in os.walk(DATA_PATH):
        for filename in filenames:
            file_path = os.path.join('/kaggle/input/', dirname, filename)
            identifiers = filename.split('.')[0].split('-')
            emotion = (int(identifiers[2]))
            if emotion == 8:
                emotion = 0
            if int(identifiers[3]) == 1:
                emotion_intensity = 'normal'
            else:
                emotion_intensity = 'strong'
            if int(identifiers[6]) % 2 == 0:
                gender = 'female'
            else:
                gender = 'male'

            data = data.append({"Emotion": emotion,
                                "Emotion intensity": emotion_intensity,
                                "Gender": gender,
                                "Path": file_path
                                },
                               ignore_index=True
                               )
    print("number of files is {}".format(len(data)))

    return data