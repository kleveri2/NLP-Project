#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:48:44 2023

@author: kleveri2
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import fasttext
import sklearn
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import numpy as np
from sklearn.ensemble import RandomForestClassifier


ds = tfds.load("huggingface:civil_comments", split = 'train')

count = 0
file1 = open("test.txt","w")

print("START")
for i in ds:
    count = count + 1
    print(count)

    txt = str(tf.keras.backend.get_value(i['text']).decode("utf-8"))
    txt2 = ""
    # Im not gappt I had to do this
    for letter in txt:
        if letter not in "`~1!2@3#4$5%6^7&8*9(0)-_=+qQwWeErRtTyYuUiIoOpP[{]}\|aAsSdDfFgGhHjJkKlL;:'zZxXcCvVbBnNmM,<.>/? ":
            continue
        txt2 = txt2 + letter
    #print(txt)
    #wrds.append(txt2)
    file1.write(txt2)

model = fasttext.train_unsupervised('test.txt',model = 'skipgram', wordNgrams = 4)

model.save_model("unigram.bin")