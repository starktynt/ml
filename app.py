from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

class API(Resource):
    def get(self , name):


        import pandas as pd
        import pickle
        df = pd.read_excel("data(1).xlsx")

        from sentence_transformers import SentenceTransformer

        try:
            model = pickle.load(open('finalized_model.sav', 'rb'))
        except:
            model = SentenceTransformer('bert-base-nli-mean-tokens')
            filename = 'finalized_model.sav'
            pickle.dump(model, open(filename, 'wb'))

        try:
            with open('parrot.pkl', 'rb') as f:
                sentence_embeddings = pickle.load(f)
        except:
            sentences = list(df["Training Phrases"])
            sentence_embeddings = model.encode(sentences)

            with open('parrot.pkl', 'wb') as f:
                pickle.dump(sentence_embeddings, f)

        from sklearn.metrics.pairwise import cosine_similarity

        def match(sente):
            XX = model.encode(sente)
            similar_vector_values = cosine_similarity([XX], sentence_embeddings)
            maxim = max(similar_vector_values[0])
            # scores we have
            scores = similar_vector_values[0]
            y = scores.tolist()

            return df.iloc[y.index(maxim)]["Intent"]
        dx = match(name)
        return {"data" :dx}

api.add_resource(API, '/bar/<string:name>')

if __name__ == '__main__':
    app.run(debug=True)
