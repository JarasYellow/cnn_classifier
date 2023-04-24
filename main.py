from dataset import dataset
from feature_extractor import preprocessing
from train_cnn import training
from model_cnn import ParallelModel, loss_fnc

data = dataset(r'/dataset/audio_speech_actors_01-24/')

n_fft = 1024
win_length = 512
hop_length = 256
SAMPLE_RATE = 48000

EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}
mel_train_chunked, mel_val_chunked, mel_test_chunked, Y_train, Y_val, Y_test = preprocessing(data, SAMPLE_RATE, n_fft, win_length, hop_length, EMOTIONS)
training(mel_train_chunked, mel_val_chunked, mel_test_chunked, ParallelModel, loss_fnc, Y_train, Y_val, Y_test, EMOTIONS)