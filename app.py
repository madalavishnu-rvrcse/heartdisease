import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    predictions=[]
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features = model.transform(final_features)
    prediction = model1.predict(final_features)
    predictions.append(prediction[0])
    prediction = model2.predict(final_features)
    predictions.append(prediction[0])
    prediction = model3.predict(final_features)
    predictions.append(prediction[0])
    output=max(predictions)
    
    if int(output)== 1: 
        prediction ='You are suffering from heart disease '
    else:
        prediction ='No heart disease found :)'
    return render_template("result.html", prediction = prediction) 

    #return render_template('index.html', prediction_text='Test report is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)