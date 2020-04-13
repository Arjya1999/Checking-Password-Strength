from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
# load the model from disk
loaded_model=pickle.load(open('checking_password_strength.pkl', 'rb'))
vectorizer=pickle.load(open('td-idf.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        password =request.form['keyword']
        data = np.array([password])
        vect = vectorizer.transform(data)
        my_prediction =loaded_model.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)