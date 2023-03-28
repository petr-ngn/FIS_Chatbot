import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
import json
import nltk
nltk.download('punkt')


from src_PN.PN_functions import update_responses, text_prep_modelling, export_files, nn_tuning

seed = 1998
np.random.seed(seed)
tf.random.set_seed(seed)

with open('./files/intents.json', 'r',encoding = "utf-8") as f:
    intents = json.load(f)

update_responses(intents)

X_train, y_train, words, classes = text_prep_modelling(intents)

export_files(intents = intents, words = words, classes = classes)

nn_model = nn_tuning(X_train, y_train, seed)

nn_model.save('NN_PN.h5')