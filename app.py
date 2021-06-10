from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Logistic_regression_model.pkl','rb'))
cv = pickle.load(open('Count_Vectorizer.pkl','rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/pred", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        review = request.form['review'].strip()
        data = [review]
        data_cv = cv.transform(data)
        prediction = model.predict(data_cv)
    
        if prediction == 1:
            return render_template('index.html', prediction_text = 'Review is Positive!')
        else :
            return render_template('index.html', prediction_text = 'Review is Negative!')
    else:
         return render_template('index.html')
if(__name__=="__main__"):
    app.run(debug=True)