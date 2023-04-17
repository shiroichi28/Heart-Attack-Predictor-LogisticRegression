import numpy as np
import pickle
from flask import Flask, request, render_template


model = pickle.load(open('heart_attack_log_model.pkl', 'rb')) 


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('Heart Disease Classifier.html')


@app.route('/predict', methods =['POST'])
def predict():
    

    features = [float(i) for i in request.form.values()]

    features = [np.array(features)]

    prediction = model.predict(features)
    
    output = prediction
    

    if output == 1:
        return render_template('Heart Disease Classifier.html', 
                               result = 'The patient is not likely to have heart disease!')
    else:
        return render_template('Heart Disease Classifier.html', 
                               result = 'The patient is likely to have heart disease!')

if __name__ == '__main__':
     app.run()
    
    
