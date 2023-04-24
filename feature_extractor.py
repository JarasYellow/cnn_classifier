import numpy as np
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
import matplotlib.pyplot as plt


def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0 ** (num_bits - 1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K
    # Generate noisy signal
    return signal + K.T * noise


def getMELspectrogram(audio, n_fft, win_length, hop_length, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=n_fft,
                                              win_length=win_length,
                                              window='hamming',
                                              hop_length=hop_length,
                                              n_mels=128,
                                              fmax=sample_rate / 2
                                              )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def splitIntoChunks(mel_spec, win_size, stride):
    t = mel_spec.shape[1]
    num_of_chunks = int(t / stride)
    chunks = []
    for i in range(num_of_chunks):
        chunk = mel_spec[:, i * stride:i * stride + win_size]
        if chunk.shape[1] == win_size:
            chunks.append(chunk)
    return np.stack(chunks, axis=0)


def preprocessing(data, SAMPLE_RATE, n_fft, win_length, hop_length, EMOTIONS):
    mel_spectrograms = []
    signals = []
    for i, file_path in enumerate(data.Path):
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5, sr=SAMPLE_RATE)
        signal = np.zeros((int(SAMPLE_RATE * 3, )))
        signal[:len(audio)] = audio
        signals.append(signal)
        print("\r Processed {}/{} files".format(i, len(data)), end='')
    signals = np.stack(signals, axis=0)

    X = signals
    train_ind, test_ind, val_ind = [], [], []
    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    for emotion in range(len(EMOTIONS)):
        emotion_ind = list(data.loc[data.Emotion == emotion, 'Emotion'].index)
        emotion_ind = np.random.permutation(emotion_ind)
        m = len(emotion_ind)
        ind_train = emotion_ind[:int(0.8 * m)]
        ind_val = emotion_ind[int(0.8 * m):int(0.9 * m)]
        ind_test = emotion_ind[int(0.9 * m):]
        X_train.append(X[ind_train, :])
        Y_train.append(np.array([emotion] * len(ind_train), dtype=np.int32))
        X_val.append(X[ind_val, :])
        Y_val.append(np.array([emotion] * len(ind_val), dtype=np.int32))
        X_test.append(X[ind_test, :])
        Y_test.append(np.array([emotion] * len(ind_test), dtype=np.int32))
        train_ind.append(ind_train)
        test_ind.append(ind_test)
        val_ind.append(ind_val)
    X_train = np.concatenate(X_train, 0)
    X_val = np.concatenate(X_val, 0)
    X_test = np.concatenate(X_test, 0)
    Y_train = np.concatenate(Y_train, 0)
    Y_val = np.concatenate(Y_val, 0)
    Y_test = np.concatenate(Y_test, 0)
    train_ind = np.concatenate(train_ind, 0)
    val_ind = np.concatenate(val_ind, 0)
    test_ind = np.concatenate(test_ind, 0)
    print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')
    print(f'X_val:{X_val.shape}, Y_val:{Y_val.shape}')
    print(f'X_test:{X_test.shape}, Y_test:{Y_test.shape}')
    # check if all are unique
    unique, count = np.unique(np.concatenate([train_ind, test_ind, val_ind], 0), return_counts=True)
    print("Number of unique indexes is {}, out of {}".format(sum(count == 1), X.shape[0]))

    del X

    # Augmentation

    aug_signals = []
    aug_labels = []
    for i in range(X_train.shape[0]):
        signal = X_train[i, :]
        augmented_signals = addAWGN(signal)
        for j in range(augmented_signals.shape[0]):
            aug_labels.append(data.loc[i, "Emotion"])
            aug_signals.append(augmented_signals[j, :])
            data = data.append(data.iloc[i], ignore_index=True)
        print("\r Processed {}/{} files".format(i, X_train.shape[0]), end='')
    aug_signals = np.stack(aug_signals, axis=0)
    X_train = np.concatenate([X_train, aug_signals], axis=0)
    aug_labels = np.stack(aug_labels, axis=0)
    Y_train = np.concatenate([Y_train, aug_labels])
    print('')
    print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')

    # Mel spectrograms

    mel_train = []
    print("Calculatin mel spectrograms for train set")
    for i in range(X_train.shape[0]):
        mel_spectrogram = getMELspectrogram(X_train[i, :], n_fft, win_length, hop_length, SAMPLE_RATE)
        mel_train.append(mel_spectrogram)
        print("\r Processed {}/{} files".format(i, X_train.shape[0]), end='')
    print('')
    del X_train

    mel_val = []
    print("Calculatin mel spectrograms for val set")
    for i in range(X_val.shape[0]):
        mel_spectrogram = getMELspectrogram(X_val[i, :], n_fft, win_length, hop_length, SAMPLE_RATE)
        mel_val.append(mel_spectrogram)
        print("\r Processed {}/{} files".format(i, X_val.shape[0]), end='')
    print('')
    del X_val

    mel_test = []
    print("Calculatin mel spectrograms for test set")
    for i in range(X_test.shape[0]):
        mel_spectrogram = getMELspectrogram(X_test[i, :], n_fft, win_length, hop_length, SAMPLE_RATE)
        mel_test.append(mel_spectrogram)
        print("\r Processed {}/{} files".format(i, X_test.shape[0]), end='')
    print('')
    del X_test

    # Get chunks
    # train set
    mel_train_chunked = []
    for mel_spec in mel_train:
        chunks = splitIntoChunks(mel_spec, win_size=128, stride=64)
        mel_train_chunked.append(chunks)
    print("Number of chunks is {}".format(chunks.shape[0]))
    # val set
    mel_val_chunked = []
    for mel_spec in mel_val:
        chunks = splitIntoChunks(mel_spec, win_size=128, stride=64)
        mel_val_chunked.append(chunks)
    print("Number of chunks is {}".format(chunks.shape[0]))
    # test set
    mel_test_chunked = []
    for mel_spec in mel_test:
        chunks = splitIntoChunks(mel_spec, win_size=128, stride=64)
        mel_test_chunked.append(chunks)
    print("Number of chunks is {}".format(chunks.shape[0]))

    del mel_train
    del mel_val
    del mel_test

    return mel_train_chunked, mel_val_chunked, mel_test_chunked, Y_train, Y_val, Y_test
