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


#Importing custom MLChatbotApp class
from src_PN.PN_ml_chatbot import MLChatbotApp

#Setting a seed for preserving reproducability
seed = 1998
np.random.seed(seed)
tf.random.set_seed(seed)

#Running the ML chatbot web appp
if __name__ == '__main__':
    app = MLChatbotApp(seed = seed)
    app.run()