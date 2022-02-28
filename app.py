from flask import Flask , send_file
from flask_restful import Resource, Api, reqparse
import nltk
'''
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
'''

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
import numpy as np
import logging
import time
import os
from gingerit.gingerit import GingerIt
from werkzeug.utils import secure_filename

import threading
import multiprocessing

import oneword
#from chat import predict_chat_2
#from chat import bot_response


app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = os.getcwd()
#app.config['MAX_CONTENT_PATH']

import logging

# Create and configure logger
logging.basicConfig(filename="ml.log",
                    format='%(asctime)s %(message)s'
                    )

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)









#translation from any language , google_Trans

translator = Translator()

logger.info("Translator setup initial call placed")



import pickle
from sentence_transformers import SentenceTransformer
try:
    with open('finalized_model.sav', 'rb') as f :
        models = pickle.load(f)
    logger.debug("Sentence transformer model not found / updated , loading again")
except:
    models = SentenceTransformer('bert-base-nli-mean-tokens')
    filename = 'finalized_model.sav'
    with open(filename, 'wb') as f :
        pickle.dump(models,f)
    logger.info("finalized_model.sav created")

logger.info("sentence transformer finalized_model loaded")

def correct(sent):

  text = sent
  result = GingerIt().parse(text)
  corrections = result['corrections']
  correctText = result['result']

  return correctText



def cleanm(text_):


    #text = correct(text)
    text = str(text_)


    logger.info("sentence tokenized , clean function call")
    text = ' '.join([w.lower() for w in word_tokenize(text)])
    text = str(text)
    text = text.lower()
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    stop = stopwords.words('english')
    logger.info("sentence stopwords removed")

    text = " ".join([word for word in text.split() if word not in stop])




    return text


##cuss inspect

from cuss_inspect import predict, predict_prob

#start = time.time()
# for simple string
#text_0 = "fuck"
#print(predict(text_0))
#print(time.time()-start)


text2trans = ""
trans_already = ""
def clean(text_):
    logger.info("function clean invoked ")
    text = text_
    text = text.lower()
    global text2trans , trans_already

    try:
        if text2trans != text:
            logger.info("setence translated under tranlate")
            text_source = translator.translate(text).src
            logger.info("Translated successfully .......")
            if text_source == "en":
                logger.info("sentence was english already")

                pass
            else :
                logger.info("sentence converted from hindi")
                text = translator.translate(text).text
                trans_already = text
        else:
            text = trans_already
    except:
        pass
    #text = correct(text)
    # text = ' '.join([w.lower() for w in word_tokenize(text)])
    # text = [word for word in text if word not in stopwords.words('english')]
    # text = re.sub('<[^<]+?>','', text)
    text = str(text)

    # text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    #stop = stopwords.words('english')

    #text = " ".join([word for word in text.split() if word not in stop])
    # stemmer = PorterStemmer()

    #wordnet_lemmatizer = WordNetLemmatizer()

    #word_tokens = nltk.word_tokenize(text)
    # word_tokens = [stemmer.stem(word) for word in word_tokens]
    #lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    #text = ' '.join(lemmatized_word)

    # text = nltk.word_tokenize(text)

    return text


logger.info("Reading dataxyx file ")
df = pd.read_excel("dataxyx.xlsx")
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)

logger.info("loading finalized_model sav")
#loaded model bert from above embeddings


logger.info("starting training process for ml se_question")

#xx = df.Intent.unique()
#z=list(df.Intent.unique())

xx = list(df['Training Phrases'])
z=list(df.Intent)
for i in range(len(xx)):
    xx[i] = cleanm(re.sub("_"," " ,xx[i]))
with open('finalized_model.sav', 'rb') as f :
    models = pickle.load(f)





sentence_embeddings = models.encode(list(xx))
y_t = []
for i in range(len(z)):
    y_t.append(i)
from sklearn.metrics.pairwise import cosine_similarity
X_train = sentence_embeddings
import numpy as np
import tensorflow as tf


try :
    logger.info("loading se _question sav")
    with open('se_question.sav','rb') as f :
        sentence_embeddings = pickle.load(f)
except :
    for i in range(len(xx)):
        xx[i] = cleanm(re.sub("_"," " ,xx[i]))
    sentence_embeddings = models.encode(list(xx))
    with open('se_question.sav','wb') as f :
        pickle.dump(sentence_embeddings , f )




'''




model_doubt_nb = GaussianNB()
y_doubt = np.array(y_t)
model_doubt_nb.fit(sentence_embeddings, y_doubt)
'''





try :
    modelt = tf.keras.models.load_model('model_answer')
except:
    inputs = tf.keras.Input(shape=(768,))
    x = tf.keras.layers.Dense(400, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(len(xx), activation=tf.nn.softmax)(x)
    modelt = tf.keras.Model(inputs=inputs, outputs=outputs)

    modelt.compile(loss="sparse_categorical_crossentropy",
                        optimizer="adam",
                        metrics=["sparse_categorical_accuracy"])


    y = np.array(y_t)
    modelt.fit(sentence_embeddings, y, epochs=75, batch_size=25)
    #filename = str(strz)


    '''
    X_train = sentence_embeddings


    logger.info("Transformer model invoked for modelt")
    import numpy as np

    import tensorflow as tf
    '''
    inputs = tf.keras.Input(shape=(768,))
    x = tf.keras.layers.Dense(400, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(len(z), activation=tf.nn.softmax)(x)
    '''
    inputs = tf.keras.Input(shape=(768,))
    outputs = tf.keras.layers.Dense(len(z), activation=tf.nn.softmax)(inputs)
    modelt = tf.keras.Model(inputs=inputs, outputs=outputs)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    modelt.compile(optimizer='Adam',loss=loss,metrics=metrics)



    y = np.array(y_t)
    '''
    with open('answer_y.sav','wb') as f:
        pickle.dump(y , f )
    #modelt.fit(sentence_embeddings, y, epochs=32, batch_size=10)

    modelt.save('model_answer')

    logger.info("model fit for doubt")

df2 = pd.read_excel('sheet.xlsx')



xxx = []

zz=[]

#df.columns
ls = []
for i in df2.columns:
  if "Tp" in str(i):
    ls.append(i)
for i in range (len(df2)):
  xxx.append(df2.IntentText[i])
  zz.append(df2.IntentID[i])
  for j in ls:
    if str(df2[j][i]) != 'nan':
      xxx.append(df2[j][i])
      zz.append(df2.IntentID[i])

try :
    with open('sheet_word.sav','rb') as f:
        sentence_embeddings = pickle.load(f)
except :
    for i in range(len(xxx)):
        xxx[i] = cleanm(re.sub("_"," " ,str(xxx[i])))
    sentence_embeddings = models.encode(list(xxx))
    with open('sheet_word.sav' , 'wb') as f :
        pickle.dump(sentence_embeddings ,f)

y_t=[]
from collections import OrderedDict
yxx = list(OrderedDict.fromkeys(zz))
for i in range(len(zz)):
    y_t.append(yxx.index(zz[i]))


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
yh = np.array(y_t)
clf.fit(sentence_embeddings, yh)


try:
    model_one2 = tf.keras.models.load_model('oneword')
except:

    ins = tf.keras.Input(shape=(768,))


    outs = tf.keras.layers.Dense(len(yxx), activation=tf.nn.softmax)(ins)
    model_one2 = tf.keras.Model(inputs=ins, outputs=outs)

    model_one2.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])


    #yh = np.array(y_t)
    model_one2.fit(sentence_embeddings, yh, epochs=32, batch_size=25)

    print("TRAINIGNG>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    model_one2.save('oneword')


def mvs_tensor(sent):

    with open('finalized_model.sav', 'rb') as f :
        models = pickle.load(f)
    cv = models.encode(clean(sent))
    #cax = models.encode(clean(sente))
    model_one2 = tf.keras.models.load_model('oneword')
    predictions = (model_one2(np.array([cv])))
    var = np.argmax(predictions, axis=1)
    #score = clf.predict_proba((np.array([cv])))[0][var]
    score = predictions[0][0].numpy()
    #print(predictions[0][0])
    #print("{:f}".format(score))
    #ind = np.argpartition(predictions , -4)
    return yxx[var[0]] , (score)


def mvs(sent):
    with open('finalized_model.sav', 'rb') as f :
        models = pickle.load(f)
    cv = models.encode(clean(sent))

    #cax = models.encode(clean(sente))
    #model_one2 = tf.keras.models.load_model('oneword')
    #predictions = (model_one2(np.array([cv])))
    #model_doubt_nb.fit(sentence_embeddings, y_doubt)
    var = clf.predict(np.array([cv]))[0]
    score = clf.predict_proba((np.array([cv])))[0][var]
    #score = predictions[0][0].numpy()
    #print(predictions[0][0])
    #print("{:f}".format(score))
    #ind = np.argpartition(predictions , -4)
    return yxx[var] , (score)


#mvs("straight line")
def match(sente) :
    with open('finalized_model.sav', 'rb') as f :
        models= pickle.load(f)
    XX = models.encode(sente)
    similar_vector_values = cosine_similarity([XX], sentence_embeddings)
    maxim = max(similar_vector_values[0])
    # scores we have
    scores = similar_vector_values[0]
    y = scores.tolist()
    return df.iloc[y.index(maxim)]["Intent"] , maxim



def match_nn(sente):
    start = time.time()

    #modelt = tf.keras.models.load_model('model_answer')
    logger.info("model prediction for entire domain started")
    with open('finalized_model.sav', 'rb') as f:
        models = pickle.load(f)

    with open('answer_y.sav' , 'rb') as f :
        y = pickle.load(f)

    logger.info("neural network called for prediction via request")
    clear = clean(sente)
    cax = models.encode(cleanm(clear))
    predictions = (modelt(np.array([cax])))
    var = np.argmax(predictions, axis=1)
    pred = z[y[var[0]]]
    score = match_cosine(sente , pred )
    ind = np.argpartition(predictions , -4)
    lis = []
    for i in ind[0][-4:]:
        lis.append(z[y[i]])


    #model_doubt_nb.fit(sentence_embeddings, y_doubt)
    #var = model_doubt_nb.predict([models.encode(cleanm(clean(sente)))])[0]
    #score = clf.predict_proba([models.encode(cleanm(clean(sente)))])[0][var]
    logger.info("domain search model successfully returned the values")
    return pred ,lis, score
    #return z[y_doubt[var]] , ["0","1","2"] , score

#req = reqparse.RequestParser()
#req.add_argument("question",type=str,required=True)


def match_cosine(sente , nas) :
    XX = models.encode(clean(sente))
    yy = models.encode(nas)
    similar_vector_values = cosine_similarity([XX], [yy])
    maxim = max(similar_vector_values[0])
    # scores we have
    #scores = similar_vector_values[0]
    #y = scores.tolist()
    return maxim

@app.route('/v1/predictnew/question',methods=["GET","POST"])
def entrynew():
    logger.info("predict/new question class called ")
    s = time.time()
    #args = req.parse_args()
    json_data = request.get_json(force=True)
    start = time.time()

    try:
        logger.info("Try block , procedding to find the answer ")

        name = str(json_data['answer'])
        ids = str(json_data['intentid'])
        if predict(name) == 1 :
            return {
                    "data": "Please do not use abusive language",
                    "score" : "NA",
                    "time" : "--"
                    }

        for i in range(len(df2['IntentID'])):
            if df2['IntentID'][i] == ids:
                ans = df2['IntentText'][i]
        score = match_cosine(name , ans)
        logger.info("attaching in the record csv")
        df = pd.read_csv("record.csv")
        # df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93}
        data = {'user_query': name,
               'query_type': 'question/doubt',
               'data_match': ids,
               'similar': 'nan',
               'similar1': 'nan',
               'similar2': 'nan',
               'time_taken': time.time() - s,
               'score': str(score)
               }
        # [name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
        df = df.append(data, ignore_index=True)
        df.to_csv("record.csv")
        return {
            "intent": ids,
            "score": str(score),
            "time": time.time()-s
        }
    except:
        logger.info("Reading record.csv")

        df = pd.read_csv("record.csv")
        # df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93}
        data = {'user_query': name,
               'query_type': 'question',
               'data_match': ids,
               'similar': 'nan',
               'similar1': 'nan',
               'similar2': 'nan',
               'time_taken': time.time() - s,
               'score': "0"
               }
        # [name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
        df = df.append(data, ignore_index=True)
        df.to_csv("record.csv")
        return {
            "data": "NAN",
            "score": "0",
            "time": time.time()-s
        }




@app.route('/v1/predict/question',methods=["GET","POST"])
def entry():
    #args = req.parse_args()
    logger.info("model entry invoked for predicting answer with intent id ")
    #try:
    json_data = request.get_json(force=True)
    start = time.time()
        #rgs = parser.parse_args()
    name = str(json_data['answer'])
    if predict(name) == 1 :
        return {
                "data": "Please do not use abusive language",
                "score" : "NA",
                "time" : "--"
                }
    dx,lis ,score = match_nn(name)
    #except:
    #    return {
    #            "data": "NAN",
    #            "time": time.time()-start,
    #            "score":"0"

    #            }
    df = pd.read_csv("record.csv")
    #df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93}
    df2 = { 'user_query':name,
            'query_type':'question',
            'data_match':str(dx),
            'similar':lis[0],
            'similar1':lis[1],
            'similar2':lis[2],
            'time_taken':time.time()-start,
            'score':score
            }
    #[name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
    df = df.append(df2, ignore_index = True)
    df.to_csv("record.csv")
    return {
                "data" :str(dx),
                "similar":lis[0],
                "similar1":lis[1],
                "similar2":lis[2],
                "score" :str(score),
                "time" : time.time()-start
                }


@app.route('/v1/train/oneword',methods=["GET","POST"])
def upload_oneword():
    import os
    try :
        #os.system('rm sheet.xlsx')
        logger.debug("model data changed , reload the app")
        os.system('rm sheet_word.sav')
        os.system('rm -rf oneword')
    except :
        pass
    f = request.files['file']
    f.save(secure_filename("sheet.xlsx"))
    return 'file uploaded successfully , Reload Server'

@app.route('/v1/train/question',methods=["GET","POST"])
def upload():
    import os
    try:
        os.system('rm dataxyx.xlsx')
        logger.debug("model data changed , reload the app")
        os.system('rm -rf model_answer')
        os.system('rm answer_y.sav')
        os.system('rm se_question.sav')
        f = request.files['file']
        f.save(secure_filename("dataxyx.xlsx"))
        return 'file uploaded successfully'
    except:
        logger.debug("try block failure for uploading question doubt")
        return ' something went wrong '

###############################################         multithreading part


def train_modelz(df , strz):
    logger.info("train model underway for  under  name of foder will be  "+str(strz))

    xx = list(df.Intent.unique())
    z=list(xx)
    '''
    xxx = []

    zz=[]

    #df.columns
    ls = []
    for i in df.columns:
        if "tp" in str(i):
            ls.append(i)
    for i in range (len(df)):
        xxx.append(df.Intent[i])
        zz.append(df.Intent[i])
    for j in ls:
        if str(df[j][i]) != 'nan':
            xxx.append(df[j][i])
            zz.append(df.Intent[i])
    for i in range(len(xxx)):
        xxx[i] = cleanm(re.sub("_"," " ,clean(xxx[i])))
    with open('finalized_model.sav', 'rb') as f :
        models = pickle.load(f)
    sentence_embeddings = models.encode(list(xxx))
    xx = xxx
    z = zz
    '''
    for i in range(len(xx)):
        xxx[i] = cleanm(re.sub("_"," " ,clean(xx[i])))
    with open('finalized_model.sav', 'rb') as f :
        models = pickle.load(f)
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
    outputs = tf.keras.layers.Dense(len(xx), activation=tf.nn.softmax)(x)
    modeltn = tf.keras.Model(inputs=inputs, outputs=outputs)

    modeltn.compile(loss="sparse_categorical_crossentropy",
                        optimizer="adam",
                        metrics=["sparse_categorical_accuracy"])


    y = np.array(y_t)
    modeltn.fit(sentence_embeddings, y, epochs=75, batch_size=25)
    filename = str(strz)

    modeltn.save("saved/"+filename)
    logger.info("model training completed for searching with parameters for doubt / question asking")
    with open(("saved/"+str(strz)+"z.sav"), 'wb') as f:
        pickle.dump(z, f)
    return








def match_nn_new(sente , micro , keysz):
    import time
    start = time.time()
    logger.info("search for doubt under paramaters invoked , starting thread")

    try:
        varz=tf.keras.models.load_model(("saved/"+str(micro)))
        logger.info("saved model found for the requested parameters")
        with open(("saved/"+ str(micro)+"z.sav") , "rb") as f:
            z = pickle.load(f)

    except:
        df = pd.read_excel('data1.xlsx')
        df3 = df.loc[df[keysz] == micro]
        if len(df3) == 0:
            return 'not found'
        else :
            train_modelz(df3 , (str(micro)))
            varz = tf.keras.models.load_model(("saved/"+str(micro)))
            with open("saved/"+(str(micro)+"z.sav") , "rb") as f:
                z = pickle.load(f)
    y_t = []
    for i in range(len(z)):
        y_t.append(i)
    y = np.array(y_t)
    cax = models.encode(((sente)))
    predictions = (varz(np.array([cax])))
    var = np.argmax(predictions, axis=1)
    similar_vector_values = cosine_similarity([models.encode((z[y[var[0]]]))],[cax])
    score = similar_vector_values[0][0]
    logger.info("search with parameters successfully returned values")
    return z[y[var[0]]], time.time() - start , score


microd = {
        "intent":"",
        "score":"",
        "time":""
        }
helper_name = ""
data_lecture = ""
data_micro = ""
data_domain = ""
logger.debug("testing ")



def micro_doubts():
    start = time.time()
    logger.info("searching with parameters in microlecture , started from threading , thread started !")

    global data_micro , helper_name
    name = helper_name
    micro = data_micro
    dx,timet , score = match_nn_new(name , str(micro) , "microlecture")
    #dx,lis ,timet , score = match_nn(name)
    #df = pd.read_csv("record.csv")
    #df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93}
    #df2 = { 'user_query':name,
    #        'query_type':'question',
    #        'data_match':str(dx),
    #        'time_taken':timet,
    #       'score':score
    #       }
    #[name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
    #df = df.append(df2, ignore_index = True)
    #df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
    #df.to_csv("record.csv",index=False)
    global microd
    microd = {
                "intent" :str(dx),
                "score" :str(score),
                "time" : time.time()-start,
                "match_type" : "microlecture"
                }
    logger.info("Thread successfully returned a value")
    return "ok"



@app.route('/vtesting/predict/question/micro',methods=["GET","POST"])
def micro_search():
    logger.info("searching with parametrs withing microsearch ,invoked from microlecture api ")
    #args = req.parse_args()
    try:
        json_data = request.get_json(force=True)
        name = str(json_data['question'])
        if predict(name) == 1 :
            return {
                    "intent": "Please do not use abusive language",
                    "score" : "NA",
                    "time" : "--"
                    }
        micro = str(json_data['microlecture'])
        dx,timet , score = match_nn_new(name , str(micro) , "microlecture")
        df = pd.read_csv("record.csv")
    #df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93}
        df2 = { 'user_query':name,
                'query_type':'question',
                'data_match':str(dx),
                'time_taken':timet,
                'score':score
            }
    #[name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
        df = df.append(df2, ignore_index = True)
        df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
        df.to_csv("record.csv",index=False)

        return {
                    "intent" :str(dx),
                    "score" :str(score),
                    "time" : timet
                }

    except:

         return {
                    "intent":"NA",
                    "score" : "NA",
                    "time" : "--"
                    }


lectured = {
        "intent":"",
        "score":"",
        "time":""
        }


def lecture_doubts():
    try:

        global data_lecture
        global helper_name
        name = helper_name
        lecture = data_lecture

        dx,timet , score = match_nn_new(name , str(lecture) , "lecture")
        global lectured
        lectured = {
                "intent" :str(dx),
                "score" :str(score),
                "time" : timet,
                "match_type" : "lecture"
                }
        return "ok"
    except:
        return "OK"
@app.route('/vtesting/predict/question/lecture',methods=["GET","POST"])
def lecture_search():
    #args = req.parse_args()
    try:
        json_data = request.get_json(force=True)
        name = str(json_data['question'])
        lecture = str(json_data['lecture'])
        dx,timet , score = match_nn_new(name , str(lecture) , "lecture")
        return {
                "intent" :str(dx),
                "score" :str(score),
                "time" : str(timet)
                }
    except:
        return {
                "data":"NA"
                }


    #rgs = parser.parse_args()

    #dx,lis ,timet , score = match_nn(name)
    ##df = pd.read_csv("record.csv")
    #df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93}
    ##df2 = { 'user_query':name,
            #'query_type':'question',
            #'data_match':str(dx),
            #'time_taken':timet,
            #'score':score
            #}
    #[name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
    #df = df.append(df2, ignore_index = True)
    #df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
    #df.to_csv("record.csv",index=Fals

domaind= {
        "intent":"NA",
        "score":"NA",
        "time":"NA"
        }


def domain_doubts():
    global data_domain , helper_name
    name = helper_name
    domain = data_domain

    dx,timet , score = match_nn_new(name , str(domain), "domain")
    global domaind
    domaind = {
           "intent" :str(dx),
           "score" :str(score),
           "time" : timet,
           "match_type" : "domain"
             }
    return "ok"

@app.route('/vtesting/predict/question/domain',methods=["GET","POST"])
def domain_search():
    #args = req.parse_args()
    try:
        json_data = request.get_json(force=True)
        name = str(json_data['question'])
        domain = str(json_data['domain'])

        dx,timet , score = match_nn_new(name , str(domain), "domain")
        return {
                "intent" :str(dx),
                "score" :str(score),
                "time" : timet
                }
    except:
        return  {
                "data" :str(dx),
                "score" :str(score),
                "time" : timet
                }



            #'data_match':str(dx),
            #'time_taken':timet,
            #'score':score
            #}
    #[name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
    #df = df.append(df2, ignore_index = True)
    #df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
    #df.to_csv("record.csv",index=Fa


#########################################################################################




##################                  ##################      ##############      multithreading
@app.route('/v1/match/answer/search',methods=["GET","POST"])
def doubt_api_final_dynamic():
    #args = req.parse_args()

    start = time.time()
    json_data = request.get_json(force=True)
    json_key = json_data.keys()
    name = str(json_data["question"])
    name = clean(name)
    #rgs = parser.parse_args()
    time.sleep(0.001)
    if predict(name) == 1 :
            return {
                    "intent":"Please do not use abusive language",
                    "score" : "NA",
                    "time" : time.time()-start
                    }
    '''
    if predict_chat_2(name) == 0 :
        return {
                "chat" : str(bot_response(name)),
                "score" : "NA",
                "time" : time.time() - start
                }
    '''
    if len(json_key) == 1 :
        try:
            dx,lis ,score = match_nn(name)
            time.sleep(0.001)
            return {
                "intent" :str(dx),
                "score" :str(score),
                "time" : time.time()-start
                }
        except:
            return{
                    "intent":"NA",
                    "time":"NA",
                    "score":"0"
                    }



    else : #len(json_key)== 2 or len(json_key)== 3 :
        time.sleep(0.01)
        global microd , lectured , domaind
        microd.clear()
        lectured.clear()
        domaind.clear()
        global data_lecture , data_domain , data_micro , helper_name
        name = str(json_data['question'])
        helper_name = name
        time.sleep(0.001)
        try:
            domain = str(json_data['domain'])
            data_domain = domain
        except:
            domain = "NA"
        time.sleep(0.001)
        try:
            lecture = str(json_data['lecture'])
            data_lecture = lecture
        except:
            lecture = "NA"
        time.sleep(0.001)
        try:
            micro = str(json_data['microlecture'])
            data_micro = micro

        except:
            micro = "NA"

        try:
            thresh = str(json_data['threshold'])
        except:
            thresh = "0.80"
        if micro != "NA":
            t1 = threading.Thread(target=micro_doubts)
            t1.start()
        time.sleep(0.001)
        if lecture !="NA":
            t2 = threading.Thread(target=lecture_doubts)
            t2.start()
        time.sleep(0.001)
        if domain !="NA":
            t3 = threading.Thread(target=domain_doubts)
            t3.start()
        time.sleep(0.001)
        if micro !="NA":
            t1.join()
            if float(microd["score"]) >= float(thresh):
                return microd
        else :
            microd = {
                        "intent":"NA",
                        "score":"NA",
                        "match_type": "microlecture",
                        "time":"NA"
                        }
        time.sleep(0.001)
        if lecture !="NA":
            t2.join()
            if float(lectured["score"]) >= float(thresh):
                return lectured
        else :
            lectured = {
                        "intent":"NA",
                        "score":"NA",
                        "match_type": "lecture",
                        "time":"NA"
                        }
        time.sleep(0.001)
        if domain != "NA":
                t3.join()
                if float(domaind["score"]) >= float(thresh):
                    return domaind
        else:
            domaind = {
                        "intent":"NA",
                        "score":"NA",
                        "match_type": "domain",
                        "time":"NA"
                        }


        '''
        t2.join()
        time.sleep(0.01)
        if float(lectured["score"]) >= float(thresh):
            #t3.terminate()
            return lectured
        t3.join()
        time.sleep(0.01)

        if float(domaind["score"]) >= float(thresh):
            return domaind


        '''
        time.sleep(0.001)
        return {
                "results": [ microd , lectured , domaind ]
                }


    #else :

        '''

        global microd ,lectured , domaind
        microd.clear()
        lectured.clear()
        domaind.clear()

        global data_lecture , data_domain , data_micro , helper_name
        name = str(json_data['question'])
        helper_name = name
        domain = str(json_data['domain'])
        data_domain = domain
        lecture = str(json_data['lecture'])
        data_lecture = lecture
        micro = str(json_data['microlecture'])
        data_micro = micro
        try:
            thresh = str(json_data['threshold'])
        except:
            thresh = "0.80"
        #microd["score"] = "0"
        #lectured["score"] = "0"
        #domaind["score"] = "0"
        #process1 = multiprocessing.Process(target=micro_doubts)
        t1 = threading.Thread(target=micro_doubts)
        #process2 = multiprocessing.Process(target=lecture_doubts)
        #process3 = multiprocessing.Process(target=domain_doubts)


        t2 = threading.Thread(target=lecture_doubts)

        t3 = threading.Thread(target=domain_doubts)
        t1.start()
        time.sleep(0.01)

        t2.start()
        time.sleep(0.01)
        t3.start()
        t1.join()
        if float(microd["score"]) >= float(thresh):
            #t2.terminate()
            #t3.terminate()
            return microd

        t2.join()
        time.sleep(0.01)
        if float(lectured["score"]) >= float(thresh):
            #t3.terminate()
            return lectured
        t3.join()
        time.sleep(0.01)

        if float(domaind["score"]) >= float(thresh):
            return domaind



        return {
                "results": [ microd , lectured , domaind ]
                }

        return {
                "intent":[microd["data"] , lectured["data"],domaind["data"]],
                "score": [microd["score"] , lectured["score"] , domaind["score"]],
                "time": time.time() - start
                }

        '''





@app.route('/v1/match/answer/searchall',methods=["GET","POST"])
def search_all():
    #args = req.parse_args()
    json_data = request.get_json(force=True)
    #rgs = parser.parse_args()
    global microd ,lectured , domaind
    microd.clear()
    lectured.clear()
    domaind.clear()

    global data_lecture , data_domain , data_micro , helper_name
    name = str(json_data['question'])
    if predict(name) == 1 :
            return {
                    "data": "Please do not use abusive language",
                    "score" : "NA",
                    "time" : "--"
                    }
    helper_name = name
    domain = str(json_data['domain'])
    data_domain = domain
    lecture = str(json_data['lecture'])
    data_lecture = lecture
    micro = str(json_data['microlecture'])
    data_micro = micro
    try:
        thresh = str(json_data['threshold'])
    except:
        thresh = "0.80"
    #microd["score"] = "0"
    #lectured["score"] = "0"
    #domaind["score"] = "0"
    #process1 = multiprocessing.Process(target=micro_doubts)
    t1 = threading.Thread(target=micro_doubts)
    #process2 = multiprocessing.Process(target=lecture_doubts)
    #process3 = multiprocessing.Process(target=domain_doubts)


    t2 = threading.Thread(target=lecture_doubts)

    t3 = threading.Thread(target=domain_doubts)
    t1.start()
    time.sleep(0.01)

    t2.start()
    time.sleep(0.01)
    t3.start()
    t1.join()
    #t2.join()
    #t3.join()

    '''
    process1.start()
    process2.start()
    process3.start()
    process1.join()
    process2.join()
    process3.join()
    '''
    if float(microd["score"]) >= float(thresh):
        #t2.terminate()
        #t3.terminate()
        return microd

    t2.join()
    if float(lectured["score"]) >= float(thresh):
        #t3.terminate()
        return lectured

    t3.join()
    if float(domaind["score"]) >= float(thresh):
        return domaind





    return  {
            "micro" : microd,
            "lecture" : lectured,
            "domain" : domaind
            }






#################



@app.route('/v1/intent/match/oneword/NB',methods=["GET","POST"])
def entrynewonewordv2():
    logger.info("verify entrynewonewordnew invoked")
    ctime = time.time()
    #args = req.parse_args()
    try:


        json_data = request.get_json(force=True)
        name = str(json_data['answer'])
        name = name.lower()
        if predict(name) == 1 :
            return {
                    "data": "Please do not use abusive language",
                    "score" : "NA",
                    "time" : "--"
                    }
        intentid = str(json_data['intentid'])
        data = oneword.oneword_res(name , intentid)
        if data["data"] == intentid:
            data['time'] = time.time()-ctime
            return data
        else :
            data["data"] = intentid
            data["score"] = "0"
            data["time"] = time.time()-ctime
            return data
    except:
        return {
                "data":"NA",
                "score":"NA",
                "time":"NA"
                }
data = {
        "question":"",
        "domain":"",
        "lecture":"",
        "microlecture":""
        }




@app.route('/v1/match/doubt/multithread',methods=["GET","POST"])
def entrynewoneword_multi():

    logger.info("verify entrynewoneword invoked")
    ctime = time.time()
    global data

    #args = req.parse_args()
    try:




        json_data = request.get_json(force=True)
        data = json_data
        name = str(json_data['answer'])
        intentid = str(json_data['intentid'])
        if predict(name) == 1 :
            search_all()
        return "neede"
    except:
        return "nan"



#mvs_tensor("straight line")
@app.route('/v1/predict/sheet/alt_tensor',methods=["GET","POST"])

def entrynewoneword_tensor():
    logger.info("verify entrynewoneword invoked")
    ctime = time.time()
    #args = req.parse_args()
    try:


        json_data = request.get_json(force=True)
        name = str(json_data['answer'])
        intentid = str(json_data['intentid'])
        if predict(name) == 1 :
            return {
                    "data": "Please do not use abusive language",
                    "score" : "NA",
                    "time" : "--"
                    }
        xs , score = mvs_tensor(clean(name))
        if xs != intentid :

            return {
                "data":intentid,
                "score":"0",
                "time":time.time()-ctime
             }
        if xs == intentid :
            return {
                "data":intentid,
                "score":"1",
                "time":time.time()-ctime
             }
    except:
        return {
                "data":"NA",
                "score":"NA",
                "time":time.time()-ctime
             }



@app.route('/v1/predict/sheet',methods=["GET","POST"])
def entrynewoneword():
    logger.info("verify entrynewoneword invoked")
    ctime = time.time()
    #args = req.parse_args()
    try:


        json_data = request.get_json(force=True)
        name = str(json_data['answer'])
        intentid = str(json_data['intentid'])
        if predict(name) == 1 :
            return {
                    "data": "Please do not use abusive language",
                    "score" : "NA",
                    "time" : "--"
                    }
        xs , score = mvs(clean(name))
        if xs != intentid :

            return {
                "data":intentid,
                "score":"0",
                "time":time.time()-ctime
             }
        score = str(score)
        df = pd.read_csv("record.csv")
        df2 = { 'user_query':name,
                'query_type':'one word',
                'data_match':xs,
                'similar':"NA",
                'similar1':"NA",
                'similar2':"NA",
                'time_taken':time.time()-ctime,
                'score':score
                }
        #[name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
        df = df.append(df2, ignore_index = True)
        df.to_csv("record.csv")
        return {
                "data":xs,
                "score":score,
                "time":time.time()-ctime
                }
    except:
        return {
            "data":"NA",
            "score":"NA",
            "time":time.time()-ctime
            }



@app.route('/v1/hello',methods=["GET"])
def ent():
    logger.info("status check requested for the ent ")
    return {
        "success": True,
        "message": "HI"
    }



@app.route('/v1/verify/answer',methods=["GET","POST"])
def entry1():
    logger.info("verify answer entry 1 invoked")
    #args = req.parse_args()
    json_data = request.get_json(force=True)
    name = str(json_data['answer'])
    if predict(name) == 1 :
            return {
                    "data": "Please do not use abusive language",
                    "score" : "NA",
                    "time" : "--"
                    }
    ctime = time.time()
    one_nn = tf.keras.models.load_model('model_answer')
    dpredictions = one_nn(np.array(models.encode([translator.translate(name).text])))
    varx = np.argmax(dpredictions, axis=1)
    ind = np.argpartition(dpredictions , -4)
    lis = []
    for i in ind[0][-4:]:
        lis.append(zx[i])
    df = pd.read_csv("record.csv")
    df2 = { 'user_query':name,
            'query_type':'one word',
            'data_match':zx[varx[0]],
            'similar':lis[0],
            'similar1':lis[1],
            'similar2':lis[2],
            'time_taken':time.time()-ctime,
            'score':"NA"
            }
    #[name,"question",str(dx) ,lis[0],lis[1],lis[2],time.time()-start,score]
    df = df.append(df2, ignore_index = True)
    df.to_csv("record.csv")
    return {
            "data":zx[varx[0]],
            "similar":lis[0],
            "similar1":lis[1],
            "similar2":lis[2],
            "time":time.time()-ctime
            }

@app.route('/v1/download/record',methods=["GET","POST"])
def entry4():
    logger.info("record csv file downloaded through api call ")

    return send_file(os.getcwd()+'/record.csv',
                     mimetype='text/csv',
                     attachment_filename='record.csv',
                     as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug = True , use_reloader =False)