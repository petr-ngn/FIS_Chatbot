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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
from bs4 import BeautifulSoup
import requests
import re
import translators as ts
import majka
import json
import pickle

def get_menza(url = "https://www.vse.cz/menza/stravovani-zizkov/"):
    
    table = BeautifulSoup(requests.get(url).content, "html.parser").find("table", class_ = "menza-table")

    foods = []

    item_foods = []

    for row in table.find_all('tr'):
        for item in row.find_all('td'):
            item_food =  str(item.find_all("div") ).replace('<div>','').replace('</div>','').replace('[', '').replace(']','')
            item_foods.append(item_food)

    for item in item_foods:
    # Split the string based on commas and numbers in parentheses
        split_items = re.split(r',\s*(?=\d+\,)|,\s*(?=\d+\))', item)
        for sub_item in split_items:
        # Remove any standalone numbers with commas
            cleaned_food_item = re.sub(r'^\d+(?=,?\s*)|\d+(?=,?\s*$)', '', sub_item)
            cleaned_food_item = re.sub(r'^[\s,]+', '', cleaned_food_item)
            foods.append(cleaned_food_item.strip())
    """
    for ind, i in enumerate(foods):
        try:
            if i[-1].isdigit():
                foods[ind] = i[:-1]
        except:
            continue
    """

    final_chalky = [i for i in foods if i != '' and len(i) > 1]

    return final_chalky

def update_responses(intents):
    for intent in intents["intents"]:
        if intent['tag'] == 'menza':
            intent['responses'] = [f"Dneska v menze je: {'; '.join(get_menza())}"]
            break

def cz_en_translate(text):
    try:
        ts.preaccelerate()
    except:
        pass
    
    return ts.translate_text(text,
                             to_language='en',
                             from_language='cs')

def majka_lemma(string):

    morph  = majka.Majka('./files/majka.w-lt')

    morph.flags = 0  # unset all flags
    morph.tags = True  # turn tag processing back on (default)
    morph.compact_tag = False  # do not return compact tag (default)
    morph.first_only = True  # return only the first entry

    try:
        return morph.find(string)[0]['lemma']
    
    except:
        return string


def text_prep_modelling(data):

    words = []
    classes = []

    X_doc = []
    y_doc = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern.lower())
            words.extend(tokens)
            X_doc.append(pattern)
            y_doc.append(intent['tag'])

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    words = sorted(set([majka_lemma(word).lower() for word in words if word not in string.punctuation]))
    classes = sorted(set(classes))

    training_data = []
    output_empty = [0] * len(classes)

    for i, doc in enumerate(X_doc):
        bag_of_words = []
        text = ' '.join([majka_lemma(word).lower() for word in doc.split(' ')])

        for word in words:
            if word in text:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(y_doc[i])] = 1
        training_data.append([bag_of_words, output_row])

    random.shuffle(training_data)
    training_data = np.array(training_data, dtype=object)# coverting our data into an array afterv shuffling

    x = np.array(list(training_data[:, 0]))# first trainig phase
    y = np.array(list(training_data[:, 1]))# second training phase

    return (x, y, words, classes)

def export_files(**kwargs):
    files_path = './files'
    for file_name, file in kwargs.items():
        format_type = 'json' if file_name == 'intents' else 'pkl'
        if format_type == 'json':
            with open(f'{files_path}/{file_name}.{format_type}', 'w', encoding = "utf-8") as f:
                json.dump(file, f, ensure_ascii = False, indent = 5)
        else:
            with open(f'{files_path}/{file_name}.{format_type}', 'wb') as f:
                pickle.dump(file, f)

def nn_tuning(X_train, y_train, seed: int):
        
    #Function for building a model and tuning its hyperparameters.
    def model_building(hp):
        
        #Input layer - with n inputs where n is the number of patterns.
        inputs = Input(shape = (len(X_train[0]), ))
        x = inputs
        
        #Tuning the number of hidden (Dense) blocks within the neural networks.
        for i in range(hp.Int('Dense_layers', min_value = 1, max_value = 10)):
            
            #Tuning the number of units, activation type, kernel and activity regularizers within each dense layer.
    
            no_units = hp.Int(f'Dense_{i}' ,min_value = 5, max_value = 512, default = 50)
            activation = hp.Choice(f'Activation_{i}', ['relu','tanh'])
            
            kernel_regularizer = L2(hp.Float(f'KernelRegularizer_{i}',
                                                       min_value = 1e-10, max_value = 0.5,
                                                       sampling = 'log'))
            
            activity_regularizer = L2(hp.Float(f'ActivityRegularizer_{i}',
                                                         min_value = 1e-10, max_value = 0.5,
                                                         sampling = 'log'))
            x = Dense(units = no_units,
                      activation = activation,
                      kernel_regularizer = kernel_regularizer,
                      activity_regularizer = activity_regularizer)(x)
        
            #Tuning the dropout rate within each hidden block
            x = Dropout((hp.Float(f'Dropout_{i}', 0.000001, 0.5, sampling='log')))(x)

        #Output layer - softmax for multiclass classification.
        outputs = Dense(len(y_train[0]), activation = 'softmax')(x)
        
        #Mapping the model's input and output layers
        model = Model(inputs, outputs, name = 'NN')
        
        #Compiling the model within Adam optimizer while minimzing binary cross entropy loss and maxizing Precision score.
        #Tuning the learning rate of the Adam optimizer
        model.compile(optimizer = Adam(hp.Float('LearningRate',
                                                                min_value = 1e-4,
                                                                max_value = 1e-2,
                                                                sampling='log')),
                      loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model
    
    #Bayesian optimization of NN while maximizing Accuracy score with 100 iterations.
    bayes_opt = BayesianOptimization(model_building,
                                          objective = 'accuracy', overwrite = False,
                                           max_trials = 100, project_name = 'Bayes_NN',
                                           seed = seed)
    
    callback = [EarlyStopping(monitor = 'accuracy', mode = 'max', patience = 7)]
    
    #Hyperparameter tuning with 200 epochs and early stopping callback after 7 epochs (if there is no improvement in the Accuracy score).
    bayes_opt.search(X_train, y_train, verbose = 0,
                 epochs = 200, callbacks = callback)
    
    #Extracting the model with the best hyperparameter values
    best_hypers = bayes_opt.get_best_hyperparameters(num_trials = 1)[0]
    final_nn = bayes_opt.hypermodel.build(best_hypers)
    
    #Final model fitting with 200 epochs and early stopping callback after 7 epochs (if there is no improvement in the Accuracy score).
    final_nn.fit(X_train, y_train, verbose = 0, epochs = 200, callbacks = callback)
    
    print("Final model hyperparameters' values:", '\n')
    for hyp_name, hyp_value in best_hypers.values.items():
        print(f'          {hyp_name}: {hyp_value}')
    
    return final_nn

def pred_class(text, vocab, labels, intents, model, threshold = 0.2):
    words_token_lemma = [majka_lemma(word.lower()) for word in nltk.word_tokenize(text)]
    bag_of_words = [0] * len(vocab)
    for w in words_token_lemma:
        for i, word in enumerate(vocab):
            if word == w:
                bag_of_words[i] = 1

    y_probs = model.predict(np.array([bag_of_words]), verbose = 0)[0]
    y_preds = [[idx, result] for idx, result in enumerate(y_probs) if result > threshold]
    y_preds.sort(key = lambda x: x[1], reverse = True)
    pred_classes = []
    for y_pred in y_preds:
        pred_classes.append(labels[y_pred[0]])
    try:
        pred_class = pred_classes[0]
        intents_list = intents['intents']
        for intent in intents_list:
            if intent['tag'] == pred_class:
                result = intent['responses'][0]
    except:
        result = "Na tuto otázku nemám odpověď. Zkuste jinou otázku."
    
    return result  