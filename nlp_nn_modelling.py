import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import nltk
nltk.download('punkt', quiet = True)
nltk.download("wordnet", quiet = True)
nltk.download('stopwords', quiet = True)

from src_PN.PN_functions import nlp_nn_modelling

seed = 1998
np.random.seed(seed)
tf.random.set_seed(seed)

nn_model_cs, words_cs, classes_cs, X_train_cs, y_train_cs = nlp_nn_modelling('cs', seed)

nn_model_en, words_en, classes_en, X_train_en, y_train_en = nlp_nn_modelling('en', seed)