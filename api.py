from flask import Flask, request, jsonify,render_template
import traceback
import numpy as np
import pickle
import re
# from sklearn.externals import joblib
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

clf = joblib.load('resume_screening_model2.pkl')
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText) 
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText


# Your API definition
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form['cancel_var']
      preprocess = cleanResume(result)
      with open('x_values1.pkl','rb') as f:
          x = pickle.load(f)
      x=np.append(x,preprocess)
      word_vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=1500)
      WordFeatures = word_vectorizer.fit_transform(x)
      answer=clf.predict(WordFeatures[962])[0]
      return render_template("home.html",result = 'It feels like this resume should belong to '+str(answer)+' departmet')

# @app.route('/test')
# def hello_world():
#     return 'Hello, World!'

# @app.route('/predict', methods=['POST'])
# def predict():
# #     if lr:
#     try:
#         lr = load_model('modelwc6.h5')
#         json_ = request.json
#         print(json_)
#         query = pd.DataFrame(json_)
#         # query = query.reindex(columns=model_columns, fill_value=0)

#         prediction = list(lr.predict_classes(query))

#         return jsonify({'prediction': str(prediction)})


#     except:

#         return jsonify({'trace': traceback.format_exc()})
if __name__ == '__main__':

    app.run()
    