from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
# import model

# intialize Flask
app = Flask(__name__)
api = Api(app)


# load trained classifier
clf_path = 'finalized_model.sav'
with open(clf_path, 'rb') as f:
    classifier = pickle.load(f)

# load trained vectorizer
vec_path = 'pickle_vectorizer.sav'
with open(vec_path, 'rb') as f:
    vectorizer = pickle.load(f)
    
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('text_content') # Input text

class RecognizeArticle(Resource):
    def get(self):
        # use parser and find the user's text_content
        args = parser.parse_args()
        user_query = args['text_content']
        # vectorize the user's text_content and make a prediction
        uq_vectorized = vectorizer.vectorizer_transform(
            np.array([user_query]))
        prediction = classifier.predict(uq_vectorized)
        pred_proba = classifier.predict_proba(uq_vectorized)
        # Output 'Negative' or 'Positive' with the score
        if prediction == 0:
            pred_text = 'Unsuitable category'
        else:
            pred_text = 'Suitable category'
            
        # round the predict proba value
        confidence = round(pred_proba[0], 3)
        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}
        
        return output

if __name__ == '__main__':
    app.run(debug=True)    