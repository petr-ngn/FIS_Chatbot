import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
import json
import pickle
from flask import Flask, render_template, request

from src_PN.PN_functions import update_responses, pred_class

with open('./files/intents.json') as json_data:
    intents = json.load(json_data)

with open('./files/words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('./files/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

nn_model = tf.keras.models.load_model('NN_PN.h5')

update_responses(intents)


seed = 1998
np.random.seed(seed)
tf.random.set_seed(seed)



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def chatbot_response():
    the_question = request.args.get('msg')
    response = pred_class(the_question, words, classes, intents, nn_model, threshold = 0.2)
    return str(response)

if __name__ == '__main__':
    app.run(host = 'localhost', debug = True) 