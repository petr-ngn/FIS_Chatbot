import numpy as np
import string
import random 
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download('omw-1.4')
nltk.download("wordnet")
tf.get_logger().setLevel('INFO')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from bs4 import BeautifulSoup
import requests
import re

from bs4 import BeautifulSoup
import requests
import re

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


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
    for ind, i in enumerate(output_list):
        try:
            if i[-1].isdigit():
                output_list[ind] = i[:-1]
        except:
            continue

    """
    
    final_chalky = [i for i in foods if i != '' and len(i) > 1]

    return final_chalky


ourData = {"intents": [
            
              {"tag": "bakalářský program",
               "patterns": ["bakalářský program", "bakalářské programy", "bakalářský studijní program", "bakalářské studijní programy"],
               "responses": ["Aplikovaná informatika; Data Analytics; Informační média a služby; Matematické metody v ekonomii; Multimédia v ekonomické praxi"]
              },
               {"tag": "magisterský program",
               "patterns": ["magisterský program", "magisterské programy", "magisterský studijní program", "magisterské studijní programy"],
               "responses": ["Informační management; Business Intelligence (IST); Data a analytika pro business; Ekonometrie a operační výzkum; Ekonomická demografie; Statistika; Business analýza (IST); Kognitivní informatika; Řízení podnikové informatiky (IST); UX výzkum a design (IST); Vývoj informačních systémů (IST); Znalostní a webové technologie,; Information Systems Management; Economic Data Analysis"]
              },

               {"tag": "magisterský předmět Data a Analytika pro business",
               "patterns": ["magisterský předmět DAB", "magisterské předměty DAB", "magisterský kurz DAB", "magisterské kurzy DAB",
                            "magisterský předmět Data a analytika pro business", "magisterské předměty Data a analytika pro business", "magisterský kurz Data a analytika pro business", "magisterské kurzy Data a analytika pro business"],
               "responses": ["Architektury business analytiky, Business a transformace businessu; Datový projekt; Informační etika, regulace a právo; Pokročilá business analytika; Řízení dat a analytiky pro business; Statistické metody pro business; Trendy a novinky v business analytice I; Trendy a novinky v business analytice II; Základní analytika a reporting"]
              },
               {"tag": "magisterský předmět Statistika",
               "patterns": ["magisterský předmět Statistika", "magisterské předměty Statistika", "magisterský kurz Statistika", "magisterské kurzy Statistika",
                            ],
               "responses": ["Časové řady; Vícerozměrná statistika; Systém národního účetnictví; Statistické úsudky; Bayesovská statistika; Ekonomie II.; Regrese; Neparametrické metody a analýzy přežívání; Teorie výběrových šetření"]
              },
              {"tag": "Ševčík",
               "patterns": ["je stále Ševčík děkanem?", "Je Ševčík stále na VŠE?"],
               "responses": ["ano, je stále tady"]
              },
              {"tag": "menza",
               "patterns": ["Co mají dneska v menze k jídlu?", "Co mají v menze na oběd?", "menza", "menze", "oběd", "stravování", "stravování v menze"],
               "responses": ['; '.join(get_menza())]
              }                    
]}


lm = WordNetLemmatizer()

ourClasses = []
newWords = []
documentX = []
documentY = []

for intent in ourData["intents"]:
    for pattern in intent["patterns"]:
        ourTokens = nltk.word_tokenize(pattern.lower())
        newWords.extend(ourTokens)
        documentX.append(pattern)
        documentY.append(intent["tag"])
    
    
    if intent["tag"] not in ourClasses:# add unexisting tags to their respective classes
        ourClasses.append(intent["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation] # set words to lowercase if not in punctuation
newWords = sorted(set(newWords))# sorting words
ourClasses = sorted(set(ourClasses))# sorting classes

trainingData = [] # training list array
outEmpty = [0] * len(ourClasses)
# BOW model

for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)
    
    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = num.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

x = num.array(list(trainingData[:, 0]))# first trainig phase
y = num.array(list(trainingData[:, 1]))# second training phase


# defining some parameters
iShape = (len(x[0]), )
oShape = len(y[0])

# the deep learning model
ourNewModel = Sequential()
ourNewModel.add(Dense(128, input_shape = iShape, activation = "relu"))
ourNewModel.add(Dropout(0.5))
ourNewModel.add(Dense(64, activation = "relu"))
ourNewModel.add(Dropout(0.3))
ourNewModel.add(Dense(oShape, activation = "softmax"))
md = tf.keras.optimizers.Adam(learning_rate = 0.01)
ourNewModel.compile(loss = 'categorical_crossentropy',
              optimizer = md,
              metrics = ["accuracy"])
ourNewModel.fit(x, y, epochs = 200, verbose = 0)


def ourText(text): 
  newtkns = nltk.word_tokenize(text)
  newtkns = [lm.lemmatize(word.lower()) for word in newtkns]
  return newtkns

def wordBag(text, vocab): 
  newtkns = ourText(text)
  bagOwords = [0] * len(vocab)
  for w in newtkns: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bagOwords[idx] = 1
  return num.array(bagOwords)

def pred_class(text, vocab, labels): 
  bagOwords = wordBag(text, vocab)
  ourResult = ourNewModel.predict(num.array([bagOwords]), verbose = 0)[0]
  newThresh = 0.2
  yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

  yp.sort(key=lambda x: x[1], reverse = True)
  newList = []
  for r in yp:
    newList.append(labels[r[0]])
  return newList

def getRes(firstlist, fJson): 
  tag = firstlist[0]
  listOfIntents = fJson["intents"]
  for i in listOfIntents: 
    if i["tag"] == tag:
      ourResult = random.choice(i["responses"])
      break
    else:
       ourResult = 'Jak to mám vědět?'
  return ourResult


from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def chatbot_response():
    the_question = request.args.get('msg')
    response = getRes(pred_class(the_question, newWords, ourClasses), ourData)
    return str(response)

if __name__ == '__main__':
    app.run(host = 'localhost', debug = True) 












