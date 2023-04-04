import os
import logging
import numpy as np
import tensorflow as tf

#Suppressing warnings and irrelevant messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
logging.getLogger('tensorflow').disabled = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Downloading NLTK packages
import nltk
nltk.download('punkt', quiet = True)
nltk.download("wordnet", quiet = True)
nltk.download('stopwords', quiet = True)
nltk.download('omw-1.4', quiet = True)

#Importing custom function for text processing and Neural Network development
from src_PN.PN_functions import nlp_nn_modelling

#Setting a seed for preserving reproducability
seed = 1998
np.random.seed(seed)
tf.random.set_seed(seed)

#Text processing, and Neural Network development for Czech language chatbot
nn_model_cs, words_cs, classes_cs, X_train_cs, y_train_cs = nlp_nn_modelling('cs', seed)

#Text processing, and Neural Network development for English language chatbot
nn_model_en, words_en, classes_en, X_train_en, y_train_en = nlp_nn_modelling('en', seed)