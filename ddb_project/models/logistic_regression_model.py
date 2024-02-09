import numpy as np
import pandas as pd
import os
import sys
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from sklearn.metrics import confusion_matrix

#import cv2 as cv
#from tensorflow.keras import datasets, layers, models
import keras
from keras.utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder


'''Task 2.1:
2.1 Let´s start with simple baseline (at your own choice).
For example, build a logistic regression model based on pre-trained word embeddings or TF-IDF vectors
of the financial news corpus **
i.e.Build a baseline model with Financial Phrasebank dataset.
What are the limitations of these baseline models?
'''