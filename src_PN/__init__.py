import numpy as np
import string
import random 
import nltk
import numpy as num 
import tensorflow as tf
import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization
import os
from bs4 import BeautifulSoup
import requests
import re
import translators as ts
import majka
import json
import pickle