from flask import Flask, render_template, request, redirect, url_for
from src_PN.PN_functions import chatbot_pred_response
import numpy as np
import tensorflow as tf
import os

class MLChatbotApp:
    def __init__(self, seed):

        #Flask app initialization
        self.app = Flask(__name__)

        #Setting path to the HTML templates
        self.app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')

        #Adding URL rules for Flask app
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/select_language', 'select_language', self.select_language, methods = ['POST'])
        self.app.add_url_rule('/czech_chatbot', 'czech_chatbot', self.czech_chatbot)
        self.app.add_url_rule('/english_chatbot', 'english_chatbot', self.english_chatbot)
        self.app.add_url_rule('/get_response', 'get_response', self.get_response, methods = ['GET'])

        #Setting a seed for preserving reproducability
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

    #Route for the web application root URL
    def index(self):
        return render_template('index.html')


    #Route for the language selection of chatbot
    def select_language(self):
        language = request.form.get('language')

        if language == 'czech':
            return redirect(url_for('czech_chatbot'))
        elif language == 'english':
            return redirect(url_for('english_chatbot'))
        else:
            return redirect(url_for('index'))


    #Route for Czech language chatbot
    def czech_chatbot(self):
        return render_template('czech_chatbot.html', language = 'czech')
    

    #Route for English language chatbot
    def english_chatbot(self):
        return render_template('english_chatbot.html', language = 'english')

    #Route for receiving user's inputs (questions) via chatbot and outputing response (based on NN predictions).
    def get_response(self):
        language = request.args.get('lang')

        if language == 'czech':
            the_question = request.args.get('msg')
            response = chatbot_pred_response(the_question, 'cs_NN')

        elif language == 'english':
            the_question = request.args.get('msg')
            response = chatbot_pred_response(the_question, 'en_NN')

        return str(response)

    #Application run
    def run(self, host = '127.0.0.1', port=5050, debug = True):
        self.app.run(host = host, port = port, debug = debug)