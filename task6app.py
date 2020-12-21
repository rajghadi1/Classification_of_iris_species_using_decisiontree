from flask import Flask, request, render_template
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('gripmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('task6.html')

@app.route('/predict',methods=['POST'])  
def predict():
   
    input_features =[float(x) for x in request.form.values()]
    features_value =[np.array(input_features)]

    output =model.predict(features_value)
    return render_template('task6.html',predict_text='Name of Iris Species is {} '.format(output))

if __name__ == '__main__':
    app.run(debug=True)