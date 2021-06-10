from flask import Flask, request, render_template
import pickle
import string
import re
from nltk import PorterStemmer, corpus

app = Flask(__name__)
model = pickle.load(open('Logistic_regression_model.pkl','rb'))

def cleanText(text):
    clean = re.compile('<.*?>')
    ps = PorterStemmer()
    stopwords = corpus.stopwords.words('english')

    html = re.sub(clean,'',text)
    p_text = ''.join([i.lower() for i in html if i not in string.punctuation])
    token = re.split('\W+',p_text)
    stop = [ps.stem(i) for i in token if i not in stopwords]
    return stop          
                       
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