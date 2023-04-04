import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
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
import matplotlib.pyplot as plt



def get_menza(url:str = "https://www.vse.cz/menza/stravovani-zizkov/") -> list:
    
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

        # Remove any standalone numbers with commas
        for sub_item in split_items:
            cleaned_food_item = re.sub(r'^\d+(?=,?\s*)|\d+(?=,?\s*$)', '', sub_item)
            cleaned_food_item = re.sub(r'^[\s,]+', '', cleaned_food_item)
            foods.append(cleaned_food_item.strip())

    final_chalky = [f'<br>• {i}' for i in foods if i != '' and len(i) > 1]

    return final_chalky



def get_transport(transport:str, lang:str) -> list:

    #API Golemio ID for transport type
    if transport == 'bus':
        awsIds = '821'
    elif transport == 'tram':
        awsIds = '172'

    #API url
    url = 'https://api.golemio.cz/v2/pid/departureboards'

    #API token
    headers = {
        'x-access-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvbGwwM0B2c2UuY3oiLCJpZCI6MTgzNSwibmFtZSI6bnVsbCwic3VybmFtZSI6bnVsbCwiaWF0IjoxNjc5MDU2MjExLCJleHAiOjExNjc5MDU2MjExLCJpc3MiOiJnb2xlbWlvIiwianRpIjoiNGRmMDczYjctMTRlMC00ZGI3LWIzMzQtNWE0ZjY4MzQxMTg4In0.hrjgSyTafjalmZMsuhIfyOt_nqoDSX3ZJJQjcMD1UiE'
            }

    #API parameters
    params = {
        'aswIds': awsIds, #transport type ID
        'minutesBefore': '0', #Looking for the upcoming departures
        'minutesAfter': '15', #Looking for the upcoming departures for the next 15 mintues
        'includeMetroTrains': 'false', #Not including metro nor trains
        'order': 'real', #Real order
        'skip': 'canceled' #Skipping canceled departures
        }

    #Translations or the transports terms for mapping.
    transport_print = {'cs': {'bus': 'Autobus', 'tram': 'Tramvaj',
                              'number': 'č.', 'direction': 'směr',
                              'departure': 'odjezd v', 'delay': 'zpoždění',
                              'no_delay': 'bez zpoždění'},

                       'en': {'bus': 'Bus', 'tram': 'Tram',
                              'number': 'no.', 'direction': 'direction',
                              'departure': 'departure at', 'delay': 'delay',
                              'no_delay': 'no delay'}}

    #Retrieve the request responses.
    response = requests.get(url, params = params, headers = headers).json()

    #List for storing the departures information.
    transports = []

    #For each departure, retrieve the information and store it in the list.
    for i in range(len(response['departures'])):

        #Transport number
        transport_number = response['departures'][i]['route']['short_name']
        #Direction of the transport (the last stop)
        last_stop = response['departures'][i]['trip']['headsign']
        #Departure time (without delay)
        arrival = response['departures'][i]['arrival_timestamp']['scheduled']
        #Delay in minutes
        delay = response['departures'][i]['delay']['minutes']

        try:
            #Extracting hours and minutes form the datetime string
            dt = datetime.datetime.fromisoformat(arrival)
            hours = dt.hour
            minutes = dt.minute
            if minutes < 10:
                minutes = f'0{minutes}'
            
            #Mapping the terms to the departure information
            transport_type = transport_print[lang]['bus'] if transport == 'bus' else transport_print[lang]['tram']
            transport_type_number = f'{transport_type} <b>{transport_print[lang]["number"]} {transport_number}</b>'
            last_stop = f'{transport_print[lang]["direction"]}: <b>{last_stop}</b>'
            arrival = f'{transport_print[lang]["departure"]} <b>{hours}:{minutes}</b>'
            delay = f'<i>{transport_print[lang]["delay"]}: {delay} min</i>' if delay > 0 else f'<i>{transport_print[lang]["no_delay"]}</i>'

            #Final transport departure information
            transport_final = f'<br>• {transport_type_number} ({last_stop}) - {arrival} ({delay})'

            #Store it to the list of transports
            transports.append(transport_final)

        #If the departure contains invalid values (when extracting the time from the datetime string), skip it   
        except:
            pass

    return transports #list of transports



def cz_en_translate(text:str) -> str:
    
    #Translating text from Czech to English
    translator = GoogleTranslator(source='cs', target='en')
    translated_text = translator.translate(text)

    return translated_text



def update_responses(intents:dict, lang:str, export:bool = True):

    if lang == 'cs':
        bus_intro = 'Ze zastávky <b>Náměstí Winstona Churchilla</b> pojedou tyto autobusové spoje v následujících 15 minutách:'
        tram_intro = 'Ze zastávky <b>Viktoria Žižkov</b> pojedou tyto tramvajové spoje v následujících 15 minutách:'
        menza_text = [f"Dneska v menze je: {' '.join(get_menza())} <br><br> Menu na další dny nalezneš zde: <a href=https://www.vse.cz/menza/stravovani-zizkov/>https://www.vse.cz/menza/stravovani-zizkov/</a>"]

    elif lang == 'en':
        bus_intro = 'These bus transports will departure from the stop <b>Náměstí Winstona Churchilla</b> in the folllowing 15 minutes:'
        tram_intro = 'These tram transports will departure from the stop <b>Viktoria Žižkov</b> in the folllowing 15 minutes:'
        menza_text = [cz_en_translate(f"Dneska v menze je: {' '.join(get_menza())} <br><br> Menu na další dny nalezneš zde: <a href=https://www.vse.cz/menza/stravovani-zizkov/>https://www.vse.cz/menza/stravovani-zizkov/</a>")]

    break_loop = 0
    
    for intent in intents.get("intents", []):

        if intent.get("tag") in [ "Bus",  "Autobus"]:
            intent["responses"] = [f"{bus_intro} {' '.join(get_transport('bus', lang))}"]
            break_loop += 1

        elif intent.get('tag') in [ "Tram",  "Tramvaj"]:
            intent["responses"] = [f"{tram_intro} {' '.join(get_transport('tram', lang))}"]

            break_loop += 1
            
        elif intent.get('tag') in ["Menza",  "Mensa", "Canteen"]:
            intent["responses"] = menza_text

            break_loop += 1

        if break_loop == 3:
            break
 
    if export: 
        with open(os.path.join('files', lang, f'{lang}_intents.json'),
                'w', encoding = "utf-8") as f:
            
            json.dump(intents, f, ensure_ascii = False, indent = 5)



def lemma_function(string:str, lang:str) -> str:

    #Lemmatization of Czech terms using Majka
    if lang == 'cs':
        
        morph = Majka(os.path.join('files', lang, 'majka.w-lt'))

        morph.flags = 0  # unset all flags
        morph.tags = True  # turn tag processing back on (default)
        morph.compact_tag = False  # do not return compact tag (default)
        morph.first_only = True  # return only the first entry

        try:
            lemmatized_text = morph.find(string)[0]['lemma']
    
        except:
            lemmatized_text = string
    
    #Lemmatization of English terms using NLTK
    elif lang == 'en':

        lm = WordNetLemmatizer()
        lemmatized_text = lm.lemmatize(string)

    return lemmatized_text



def text_cleaning_tokens(text:str, lang:str) -> list:

    #Reading stop words
    if lang == 'cs':
        
        with open(os.path.join('files', lang, 'stop_words_czech.json'),
                  'r', encoding = "utf-8") as f:
            
            stop_words = json.load(f)
    
    elif lang == 'en':
        stop_words = stopwords.words('english')
    
    #Tokenization
    tokens = nltk.word_tokenize(text)

    #Normalization
    tokens = [word if word.isupper() else word.lower() for i, word in enumerate(tokens)]

    #Removing punctuations
    tokens = [word for word in tokens if word not in string.punctuation]

    #Lemmatization
    tokens = [word if word.isupper() else lemma_function(word, lang) for word in tokens]

    #Removing stop words
    tokens = [word for word in tokens if word not in stop_words]

    return tokens #Final tokenized and cleaned text



def text_prep_modelling(data:dict, lang:str, export:bool = True) -> tuple:

    words = []
    classes = []

    X_doc = []
    y_doc = []

    #For each intent's pattern, tokenize and the pattern, assign it to the words, training data including its respective label
    for intent in data['intents']:
        for pattern in intent['patterns']:
            tokens = text_cleaning_tokens(pattern, lang)
            words.extend(tokens)
            X_doc.append(pattern)
            y_doc.append(intent['tag'])

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    #Unique and sorted words and classes
    words = sorted(set(words))
    classes = sorted(set(classes))

    #Training data of bag of words and labels
    training_data = []
    output_empty = [0] * len(classes)

    for i, doc in enumerate(X_doc):

        bag_of_words = []

        text =  ' '.join(text_cleaning_tokens(doc, lang))

        for word in words:
            if word in text:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(y_doc[i])] = 1
        training_data.append([bag_of_words, output_row])

    #Shuffle the training data
    random.shuffle(training_data)
    training_data = np.array(training_data, dtype = object)

    #Training data as an input for NN modelling
    x = np.array(list(training_data[:, 0]))
    y = np.array(list(training_data[:, 1]))

    #Exporting the training data, words and classes
    if export:
        export_files = [x, y, words, classes]
        export_file_names = ['X_train', 'y_train', 'words', 'classes']

        for file, file_name in zip(export_files, export_file_names):

            with open(os.path.join('files', lang,
                                   f'{lang}_{file_name}.pkl'),
                    'wb') as f:
                
                pickle.dump(file, f)

    return (x, y, words, classes)



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        tp_fn = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall_score = tp / (tp_fn + K.epsilon())

        return recall_score

    def precision(y_true, y_pred):

        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        tp_fp = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision_score = tp / (tp_fp + K.epsilon())

        return precision_score

    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)

    f1_score = 2 * ((precision_score * recall_score) / (precision_score + recall_score + K.epsilon()))

    return f1_score



def categorical_focal_loss_function(alpha:float, gamma:float):

    """
    Focal Loss = - alpha * (1 - p) ^ {gamma} * y * log(p)
    """

    def categorical_focal_loss(y_true, y_pred):
        
        #Clipping the predicted probabilities to avoid NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        #Converting the true classes into float (can't be integers)
        y_true = K.cast(y_true, dtype = 'float32')

        #Individual focal losses
        focal_loss = - alpha * K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred)

        final_categorical_focal_loss = K.mean(
                                                K.sum(focal_loss, axis = -1)
                                              )

        return final_categorical_focal_loss

    return categorical_focal_loss



def focal_loss_f1_plot(history, lang:str):

    title_name = "Czech" if lang == 'cs' else "English"

    fig = plt.figure(figsize=(9, 7))

    #F1 score
    ax1 = plt.gca()
    ax1.plot(history.history['f1'], color='blue', label='Training F1 score')
    ax1.set_ylabel('F1 score', fontsize=13)

    #Focal loss
    ax2 = ax1.twinx()
    ax2.plot(history.history['loss'], color='red', label='Training Focal Loss')
    ax2.set_ylabel('Focal Loss', fontsize=13)

    ax1.yaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

    plt.title(f"{title_name} NN's Model Focal Loss and F1 score", fontsize=15)
    ax1.set_xlabel('Epochs', fontsize=13)
    ax1.xaxis.set_tick_params(labelsize=12)

    plt.grid(linestyle='--')

    lines = ax1.get_lines() + ax2.get_lines()

    plt.legend(lines, [line.get_label() for line in lines],
               loc = 'upper center', fontsize = 13,
               bbox_to_anchor = (0.5, -0.09), ncol = 2)

    plt.tight_layout()

    plt.savefig(os.path.join('models', lang, f'{lang}_NN_FocalLoss_F1_plot.jpg'), dpi = 1200)

    #plt.show()



def nn_tuning(X_train:np.array, y_train:np.array, seed:int, name:str):
        
    #Function for building a model and tuning its hyperparameters.
    def model_building(hp):
        
        #Input layer - with n inputs (n = number of features).
        inputs = Input(shape = (len(X_train[0]), ), name = 'InputLayer')
        x = inputs
        
        #Tuning the number of hidden blocks within the neural networks.
        for i in range(hp.Int('Dense_layers', min_value = 1, max_value = 10)):
            
            #Tuning the number of units, activation type, kernel and activity regularizers within each dense layer.

            no_units = hp.Int(f'DenseLayer_{i}' ,min_value = 5, max_value = 512, default = 50)

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

            dropout_rate = (hp.Float(f'Dropout_{i}', 0.000001, 0.5, sampling='log'))

            x = Dropout(dropout_rate)(x)



        #Output layer - softmax for multiclass classification - with n outputs where n is the number of classes
        outputs = Dense(len(y_train[0]), activation = 'softmax', name = 'OutpuLayer')(x)
        
        #Mapping the model's input and output layers
        model = Model(inputs, outputs, name = 'NN_PN')
        
        #Tuning the learning rate of the Adam optimizer
        learning_rate = hp.Float('LearningRate',
                                min_value = 1e-4,
                                max_value = 1e-2,
                                sampling='log')
        
        #Tuning the Alpha parameter of the Categorical Focal Loss
        alpha = hp.Float('Alpha',
                         min_value = 0.01, max_value = 10,
                         sampling = 'log')
        
        #Tuning the Gamma parameter of the Categorical Focal Loss
        gamma = hp.Float('Gamma',
                         min_value = 0.01, max_value = 5,
                         sampling = 'log')
        
        #Compiling the model within Adam optimizer while minimizing Categorical Focal Loss and maxizing F1 score
        model.compile(optimizer = Adam(learning_rate = learning_rate),
                      metrics = [f1],
                      loss = categorical_focal_loss_function(alpha = alpha,
                                                             gamma = gamma)
                      )

        return model
    
    #Bayesian optimization of NN while minimizing Categorical Focal Loss with 100 iterations
    bayes_opt = BayesianOptimization(model_building,
                                          objective = 'loss',
                                          overwrite = True,
                                           max_trials = 100,
                                           project_name = f'Bayes_NN_{name}',
                                           seed = seed)
    
    #Early stopping after 7 epochs while monitoring the Categorical Focal Loss
    callback = [EarlyStopping(monitor = 'loss', mode = 'min', patience = 7)]
    
    

    #Hyperparameter tuning with 200 epochs and early stopping callback after 7 epochs (if there is no improvement in the Categorical Focal Loss)
    sys.stderr = open(os.devnull, 'w') #Suppressing the tracebacks of Bayesian Optimization
    bayes_opt.search(X_train, y_train, verbose = 0,
                 epochs = 200, callbacks = callback)
    sys.stderr = sys.__stderr__ #Restoring the tracebacks

    #Extracting the model with the best hyperparameter values
    best_hypers = bayes_opt.get_best_hyperparameters(num_trials = 1)[0]
    final_nn = bayes_opt.hypermodel.build(best_hypers)

    #Saving the best hyperparameters in a json file
    with open(os.path.join('models', name, f'{name}_best_hyperparameters.pkl'),
             'wb') as f:
        pickle.dump(best_hypers, f)

    #Removing the Bayesian Optimization folder from the directory
    shutil.rmtree(os.path.join(f'Bayes_NN_{name}'),
                  ignore_errors = True)
    
    #Final model fitting with 200 epochs and early stopping callback after 7 epochs (if there is no improvement in the Categorical Focal Loss)
    history = final_nn.fit(X_train, y_train, verbose = 0,
                            epochs = 200, callbacks = callback)
    
    #Saving the training history of focal loss and F1 score as a plot
    focal_loss_f1_plot(history, name)

    #Saving the final model structure as a plot
    tf.keras.utils.plot_model(final_nn, show_shapes = True, show_layer_activations = True,
                          show_layer_names = True, expand_nested = True,
                          to_file = os.path.join('models', name, f'{name}_NN_plot.jpg'))
    
    return final_nn #Final NN model



def h5_to_tflite_converter(nn_model, model_name:str):

    #Saving the NN model in h5 format
    nn_model.save(os.path.join('models', model_name.split('_')[0],
                               f'{model_name}.h5'))
    
    #Loading the NN model h5 format
    h5_model = load_model(os.path.join('models', model_name.split('_')[0],
                                       f'{model_name}.h5'),

                          custom_objects = {'categorical_focal_loss': categorical_focal_loss_function,
                                            'f1': f1}
                        )
    
    #Converting the NN model from h5 to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
    tflite_model = converter.convert()

    #Exporting the TFLite model in tflite format
    
    open(os.path.join('models', model_name.split('_')[0],
                      f'{model_name}.tflite'),
        "wb").write(tflite_model)

    return tflite_model



def print_statement_title(text:str, no_dashes:int = 100):
        no_dashes_left = int((no_dashes - len(text) - 1)/2)*"-"
        no_dashes_right  = f"{(no_dashes - len(text) - len(no_dashes_left) - 2)*'-'}"
        print(no_dashes * "-")
        print(f'{no_dashes_left} {text} {no_dashes_right}')
        print(no_dashes * "-", '\n')

def print_statement_step(text:str, indent:int = 10):
    print(f'{" "*indent}{text}')



def nlp_nn_modelling(lang:str, seed:int):

    print_statement_title(f'{lang.upper()} NLP NEURAL NETWORK MODELLING')

    print_statement_step('STARTING NLP NEURAL NETWORK MODELLING', 0)
    
    print('\n')


    print_statement_step('1. Loading intents')


    with open(os.path.join('files', lang,
                            f'{lang}_intents.json'),
            'r', encoding = "utf-8") as f:
        
        intents = json.load(f)


    print_statement_step("2. Updating intents' responses")
    
    update_responses(intents, lang)


    print_statement_step('3. Text preprocessing and exporting')
    
    x, y, words, classes = text_prep_modelling(intents, lang)


    print_statement_step('4. Bayesian Optimization and Neural Network modelling')
    nn_model = nn_tuning(x, y, seed, lang)


    print('\n')
    print_statement_step('5. Converting the NN model to TFLite')

    nn_model_final = h5_to_tflite_converter(nn_model, f'{lang}_NN')

    print('\n')
    print_statement_step('NLP NEURAL NETWORK MODELLING FINISHED', 0)
    print('\n')

    return (nn_model_final, words, classes, x, y)



def chatbot_pred_response(text:str, model_name:str, threshold:float = 0.2) -> str:

    #Loading the intents, words and classes which are related to given NN model
    if 'cs' in model_name:
        lang = 'cs'
    elif 'en' in model_name:
        lang = 'en'

    with open(os.path.join('files', lang, f'{lang}_words.pkl'),
                      'rb') as f:     
        words = pickle.load(f)

    with open(os.path.join('files', lang, f'{lang}_classes.pkl'),
                      'rb') as f:
        classes = pickle.load(f)

    #Cleaning and tokenization of the text input
    tokens = text_cleaning_tokens(text, lang)

    with open(os.path.join('files', lang, f'{lang}_intents.json'),
                    'r', encoding = "utf-8") as f:
        intents = json.load(f)

    #Updating intents' responses (specifically canteen and public transport responses)
    update_responses(intents, lang)

    #Bag of words input for the NN model
    bag_of_words = [0] * len(words)

    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bag_of_words[i] = 1

    #Loading the TFLite NN model
    interpreter = tf.lite.Interpreter(model_path = os.path.join('models', lang, f'{model_name}.tflite'))
    
    #Allocating the necessary memory for the model's input and output tensors
    interpreter.allocate_tensors()

    #Mapping the bag of words to the NN's input layer.
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                           np.array([bag_of_words]).astype('float32'))
    
    #Performing the inference on the input tensor using the loaded NN model
    interpreter.invoke()

    #Retrieving the output tensor (predicted probabilites of intents)
    y_probs = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    #Flattening the output tensor into 1D array
    y_probs = np.array(y_probs).flatten()

    #Filtering the significant predicted probabiliites and sorting them in descending order
    y_preds = [[idx, result] for idx, result in enumerate(y_probs) if result > threshold]
    y_preds.sort(key = lambda x: x[1], reverse = True)

    #Accessing the predicted intents' tags based on the probabilities' indices
    pred_classes = [classes[y_pred[0]] for y_pred in y_preds]

    #Accessing the intent's response whose significant probability is the highest
    try: 
        pred_class = pred_classes[0]
        intents_list = intents['intents']
        for intent in intents_list:
            if intent['tag'] == pred_class:
                result = intent['responses'][0]
    
    #If there is no significant probability, the NN model returns a default response
    except:
        if 'cs' in model_name:
            result = "Na tuto otázku nemám odpověď. Zkus jinou otázku."

        elif 'en' in model_name:
            result = "I don't have an answer to this question. Try another question."
    
    return result #responose of the chatbot