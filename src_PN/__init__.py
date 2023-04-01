import os
import numpy as np
import shutil
import datetime
import string
import random 
import nltk
import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras_tuner.tuners import BayesianOptimization
from bs4 import BeautifulSoup
import requests
import re
from majka import Majka
import json
import pickle
from deep_translator import GoogleTranslator
from keras import backend as K
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer