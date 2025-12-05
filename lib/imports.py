# common imports used across the project

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from time import time
import os
import hashlib

#importing all the machine learning python libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Masking, LeakyReLU, TimeDistributed, Conv1D, BatchNormalization
from keras.layers import GlobalAveragePooling1D, Dropout, MaxPool1D, Flatten, Input, MaxPooling1D, GRU
from keras import Model
from keras.optimizers import Adam, Nadam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore[import] (doesn't recognize keras even if installed)
from keras.preprocessing.sequence import pad_sequences # type: ignore[import]