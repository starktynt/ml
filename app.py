from flask import Flask
from flask_restful import Resource, Api, reqparse
import nltk
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
import time
from sklearn.tree import DecisionTreeClassifier

import pandas as pd 
from googletrans import Translator
from flask import request
import pickle

import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import logging
import time

from gingerit.gingerit import GingerIt
from werkzeug.utils import secure_filename


app = Flask(__name__)
api = Api(app)



translator = Translator()



try:
        models = pickle.load(open('finalized_model.sav', 'rb'))
except:
        models = SentenceTransformer('bert-base-nli-mean-tokens')
        filename = 'finalized_model.sav'
        pickle.dump(models, open(filename, 'wb'))

try:
    df = pd.read_csv(data_one.csv)
except:
    df = pd.read_excel("data.xlsx")
    df = df[:10]
    #df.head(10)
    list_col = []
    
    for i in df.columns:
        list_col.append(i)
    for i in list_col:
        for j in range(len(df[i])):
            df[i][j] = translator.translate(df[i][j]).text
        #print(x)
    df.to_csv("data_one.csv")
        

try:
         with open('one_word.pkl', 'rb') as f:
             yy = pickle.load(f)
except:
         data = {
                 "text":[],
                 "intent":[]
                 }
         for i in list_col:
             for j in range((len(df[i]))):
                 if i =="intent_text":
                     pass
                 else:
                     data["text"].append(df[i][j])
                     data["intent"].append(df["intent_text"][j])
         df1 = pd.DataFrame(data)

                 
         yy = models.encode(list(df1.text))
         with open('one_word.pkl', 'wb') as f:
             pickle.dump(yy, f)


try:
    df1 = pd.read_csv("data.csv")
except:
    
    data = {
            "text":[],
            "intent":[]
             }
    for i in list_col:
        for j in range((len(df[i]))):
            if i=="intent_text":
                pass
            else:
                data["text"].append(df[i][j])
                data["intent"].append(df["intent_text"][j])
        df1= pd.DataFrame(data)
        df1.to_csv("data.csv")
#df1= pd.DataFrame(data)
zx = df1.intent.unique()
zx = list(zx)
try:
        with open("dtree.sav","rb") as f:
          tree_model = pickle.load(f)

except:
        data = {
                 "text":[],
                 "intent":[]
             }
        for i in list_col:
            for j in range((len(df[i]))):
                    if i=="intent_text":
                        pass
                    else:
                        data["text"].append(df[i][j])
                        data["intent"].append(df["intent_text"][j])
        df1= pd.DataFrame(data)
        zx = df1.intent.unique()
        zx = list(zx)
        for i in range(len(df1)):
            df1.intent[i] = zx.index(df1.intent[i])
            X_train = yy
        y_train = list(df1.intent)
        dtree_model = DecisionTreeClassifier(max_depth = 10).fit(X_train, y_train)
        filename = 'dtree.sav'
        with open(filename,"wb") as f:
          pickle.dump(dtree_model, f)


def correct(sent):

  text = sent
  result = GingerIt().parse(text)
  corrections = result['corrections']
  correctText = result['result']

  return correctText

def clean(text_):
    text = text_
    text = correct(text)
    # text = ' '.join([w.lower() for w in word_tokenize(text)])
    # text = [word for word in text if word not in stopwords.words('english')]
    # text = re.sub('<[^<]+?>','', text)
    text = text.lower()
    # text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    stop = stopwords.words('english')

    text = " ".join([word for word in text.split() if word not in stop])
    # stemmer = PorterStemmer()

    wordnet_lemmatizer = WordNetLemmatizer()

    word_tokens = nltk.word_tokenize(text)
    # word_tokens = [stemmer.stem(word) for word in word_tokens]
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    text = ' '.join(lemmatized_word)

    # text = nltk.word_tokenize(text)

    return text



df = pd.read_excel("data(1).xlsx")
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)


try:
    model = pickle.load(open('finalized_model.sav', 'rb'))
except:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))


xx = df.Intent.unique()
z=list(xx)
for i in range(len(xx)):
   xx[i] = clean(re.sub("_"," " ,xx[i]))


sentence_embeddings = models.encode(list(xx))
y_t = []
for i in range(len(z)):
    y_t.append(i)


from sklearn.metrics.pairwise import cosine_similarity

X_train = sentence_embeddings



import numpy as np

import tensorflow as tf

inputs = tf.keras.Input(shape=(768,))
x = tf.keras.layers.Dense(400, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(71, activation=tf.nn.softmax)(x)
modelt = tf.keras.Model(inputs=inputs, outputs=outputs)

modelt.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])


y = np.array(y_t)
modelt.fit(sentence_embeddings, y, epochs=75, batch_size=25)


# filename = 'tensorflow_sentence.sav'


def match(sente) :
    XX = model.encode(sente)
    similar_vector_values = cosine_similarity([XX], sentence_embeddings)
    maxim = max(similar_vector_values[0])
    # scores we have
    scores = similar_vector_values[0]
    y = scores.tolist()
    return df.iloc[y.index(maxim)]["Intent"] , maxim

def match_nn(sente):
    start = time.time()
    cax = models.encode(clean(sente))
    predictions = (modelt.predict(np.array([cax])))
    var = np.argmax(predictions, axis=1)
    return z[y[var[0]]] , time.time() - start

#req = reqparse.RequestParser()
#req.add_argument("question",type=str,required=True)

@app.route('/v1/predict/question',methods=["GET","POST"])
def entry():
	    #args = req.parse_args()
	    json_data = request.get_json(force=True)
	    
	    start = time.time()
	    #args = parser.parse_args()
	    name = str(json_data['question'])
	    dx,score = match_nn(name)
	    return {
	        "data" :str(dx),
	        "score" :str(score),
	        "time" : time.time()-start
	        
	    }
	    
	    
@app.route('/v1/train/question',methods=["GET","POST"])
def upload():
    import os
    
    os.system('rm data(1).xlsx')
    os.system('rm parrot.pkl')
    os.system('rm final.sav')
    f = request.files['file']
    f.save(secure_filename("data(1).xlsx"))
    return 'file uploaded successfully'
'''
class API(Resource):
	    
	def get(self):

	    #args = req.parse_args()
	    json_data = request.get_json(force=True)
	    
	    start = time.time()
	    #args = parser.parse_args()
	    name = str(json_data['question'])
	    dx,score = match_ml(name)
	    return {
	        "data" :str(dx),
	        "score" :str(score),
	        "time" : time.time()-start
	        
	    }
	
'''

@app.route('/v1/verify/answer',methods=["GET","POST"])
def entry1():
	    #args = req.parse_args()
	    json_data = request.get_json(force=True)	    
	    name = str(json_data['answer'])	    
	    ctime = time.time()
	    dtree_predictions = dtree_model.predict(models.encode([translator.translate(name).text]))
	    cx = zx[dtree_predictions[0]]
	    return {"data":cx,
                "time":time.time()-ctime
                }
'''
class H(Resource):
        def get(self ):
            parser = reqparse.RequestParser()
            args = parser.parse_args()


#api.add_resource(API, '/v1/predict/question')
api.add_resource(H,'/v1/verify/answer')
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0')

xx = df.Intent.unique()
for i in range(len(xx)):
  xx[i] = clean(re.sub("_"," " ,xx[i]))
z=list(xx)
z
sentence_embeddings = models.encode(z)
y_t = []
for i in range(len(z)):
  y_t.append(i)
