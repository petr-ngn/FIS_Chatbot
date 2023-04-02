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
nltk.download('omw-1.4', quiet = True)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src_PN.PN_functions import pred_class
from flask import Flask, render_template, request, redirect, url_for

seed = 1998
np.random.seed(seed)
tf.random.set_seed(seed)



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/select_language', methods = ['POST'])
def select_language():
    language = request.form.get('language')
    
    if language == 'czech':
        return redirect(url_for('czech_chatbot'))
    elif language == 'english':
        return redirect(url_for('english_chatbot'))
    else:
        return redirect(url_for('index'))


@app.route('/czech_chatbot')
def czech_chatbot():
    return render_template('czech_chatbot.html', language = 'czech')


@app.route('/english_chatbot')
def english_chatbot():
    return render_template('english_chatbot.html', language = 'english')


@app.route('/get_response', methods = ['GET'])
def get_response():
    
    language = request.args.get('lang')
    
    if language == 'czech':
        the_question = request.args.get('msg')
        response = pred_class(the_question, 'cs_NN')

    elif language == 'english':
        the_question = request.args.get('msg')
        response = pred_class(the_question, 'en_NN')

    return str(response)


if __name__ == '__main__':
    app.run(port = 5050, debug = True)