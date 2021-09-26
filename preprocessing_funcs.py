##import files
from scipy import ndimage
import os
import sys
import re
import librosa  # python package for music and audio analysis
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.utils import class_weight
import seaborn as sns
import skimage.transform as st


def segment_audio(signal, is_sig=True, sig_thresh=3, noise_thresh=2.5, plot=True):
    """function that takes an audio file and
    returns either signal segments or noise segments

    Args:
    signal (np array): sound signal
    is_sig (bool): choose True for sig, False for noise
    sig_thresh (float): filter out values 
                    that are sig_thresh*row and col medians
    noise_thresh (float): can be different than sig_thresh
    plot (bool): show filter plots or off

    Returns 
    1) numpy array of bool flag 
    2) segmented numpy array of filtered audio
    """
    spec = np.abs(librosa.stft(signal))  # calculate spectrogram
    mat = spec/np.max(spec)  # normalize by max value

    col_medians = np.median(mat, axis=0)
    row_medians = np.median(mat, axis=1)

    if is_sig == True:
        thresh = sig_thresh
    else:
        thresh = noise_thresh

    row_mat = mat >= thresh*row_medians.reshape(-1, 1)
    col_mat = mat >= thresh*col_medians.reshape(1, -1)
    filter_1 = (row_mat & col_mat).astype(int)  # filter mask
    # apply erosion filter to get rid of spurious pixels (island 1s)
    erosion_filter = ndimage.binary_erosion(
        filter_1, structure=np.ones((4, 4))).astype(filter_1.dtype)
    # apply dilation to smoothen
    dilation_filter = ndimage.binary_dilation(
        erosion_filter, structure=np.ones((4, 4))).astype(erosion_filter.dtype)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        librosa.display.specshow(filter_1, y_axis='linear', ax=ax1)
        librosa.display.specshow(erosion_filter, y_axis='linear', ax=ax2)
        librosa.display.specshow(dilation_filter,  y_axis='linear', ax=ax3)
        plt.show()

    # find columns that have more than 1 ones
    temp_filter = np.sum(dilation_filter, axis=0)
    if is_sig == True:
        column_mask = (temp_filter >= 1).reshape(-1, 1)
    else:
        column_mask = (temp_filter < 1).reshape(-1, 1)
    # smoothen twice with dilation filter
    column_mask = ndimage.binary_dilation(
        column_mask, structure=np.ones((4, 1))).astype(column_mask.dtype)
    column_mask = ndimage.binary_dilation(
        column_mask, structure=np.ones((4, 1))).astype(column_mask.dtype)

    out_bool = st.resize(column_mask, (signal.shape[0], 1)).astype(bool)
    out_sig = signal[out_bool[:, 0]].copy()

    return out_bool, out_sig


def split_into_chunks(spec, fname, step_size=512, bird_name='aldfly'):
    """function to break signal into chunks

    Args:
    spec (np matrix): spectrogram
    fname (int): file name for identification later
    step_size (int): len of chunk
    bird_name (str): which bird is it from the mapping or noise

    Returns numpy split array for X_train, y_train
    """
    l, w = spec.shape
    num_chunks = int(w/step_size)  # total full sized chunks
    # zero pad the last incomplete chunk if it exists
    zero_pad_flag = 0
    if num_chunks < w/step_size:
        num_chunks += 1
        zero_pad_flag = 1

    # initialize zero arrays
    X_train = np.zeros(shape=(num_chunks, l, step_size))
    y_train = np.zeros(shape=(num_chunks))
    file_name = np.zeros(shape=(num_chunks))

    lower, upper = 0, step_size
    i = 0  # index of num_chunks
    while i<num_chunks:
      # zero padded chunk
      if i == num_chunks-1 and zero_pad_flag==1:
        #####################check constant value
        X_train[i] = np.pad(spec[:,lower:], ((0,0),(0,step_size-(w-lower))), 'constant',constant_values=(-80))
      # regular chunk
      else:
        X_train[i] = spec[:,lower:upper]
        
      y_train[i] = bird_dict[bird_name] #for aldfly it is 0, mapping
      file_name[i] = fname
      i+=1
      upper+=step_size
      lower+=step_size

    return X_train, y_train, file_name


def create_train_samples(bird_dir):
    bird_files = [f for f in os.listdir(bird_dir) if not f.startswith('.')]
    #bird_files = bird_files[1:4].copy()
    i = 0
    print(len(bird_files))
    if len(bird_files) > 0:
        SOUND_DIR = bird_dir + bird_files[i]
        fname = int(re.findall(r'\d+', bird_files[i])[0])

        signal, sr = librosa.load(SOUND_DIR, duration=48000)
        b_bird, bird_sig = segment_audio(signal, plot=False)
        b_noise, noise = segment_audio(signal, is_sig=False, plot=False)
        bird_spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(bird_sig)), ref=np.max)
        noise_spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(noise)), ref=np.max)
        X, y, f = split_into_chunks(
            bird_spec, fname=fname, bird_name=bird_dir.split('/')[1])
        X1, y1, f1 = split_into_chunks(
            noise_spec, fname=fname, bird_name='noise')

    for i in range(1, len(bird_files)):
        print(i)
        SOUND_DIR = bird_dir + bird_files[i]
        fname = int(re.findall(r'\d+', bird_files[i])[0])

        signal, sr = librosa.load(SOUND_DIR, duration=48000)
        b_bird, bird_sig = segment_audio(signal, plot=False)
        b_noise, noise = segment_audio(signal, is_sig=False, plot=False)
        bird_spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(bird_sig)), ref=np.max)
        noise_spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(noise)), ref=np.max)
        X_temp, y_temp, f_temp = split_into_chunks(
            bird_spec, fname, bird_name=bird_dir.split('/')[1])
        X1_temp, y1_temp, f1_temp = split_into_chunks(
            noise_spec, fname, bird_name='noise')
        X = np.concatenate((X, X_temp))
        X1 = np.concatenate((X1, X1_temp))
        y = np.concatenate((y, y_temp))
        y1 = np.concatenate((y1, y1_temp))
        f = np.concatenate((f, f_temp))
        f1 = np.concatenate((f1, f1_temp))

    return X, y, f, X1, y1, f1
