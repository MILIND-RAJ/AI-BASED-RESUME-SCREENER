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
def find_prof(num):
    if num==0:
        return 'Advocate'
    elif num==1:
        return 'Arts'
    elif num==2:
        return 'Automation Testing'
    elif num==3:
        return 'Blockchain'
    elif num==4:
        return 'Business Analyst'
    elif num==5:
        return 'Civil Engineer'
    elif num==6:
        return 'Data Science'
    elif num==7:
        return 'Database'
    elif num==8:
        return 'DevOps Engineer'
    elif num==9:
        return 'DotNet Developer'
    elif num==10:
        return 'ETL Developer'
    elif num==11:
        return 'Electrical Engineering'
    elif num==12:
        return 'HR'
    elif num==13:
        return 'Hadoop'
    elif num==14:
        return 'Health and fitness'
    elif num==15:
        return 'Java Developer'
    elif num==16:
        return 'Mechanical Engineer'
    elif num==17:
        return 'Network Security Engineer'
    elif num==18:
        return 'Operations Manager'
    elif num==19:
        return 'PMO'
    elif num==20:
        return 'Python Developer'
    elif num==21:
        return 'SAP Developer'
    elif num==22:
        return 'Sales'
    elif num==23:
        return 'Testing'
    elif num==24:
        return 'Web Designing'
    



# Your API definition
app = Flask(__name__,static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

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
      answer=find_prof(clf.predict(WordFeatures[962])[0])
      return render_template("index.html",result = 'It feels like this resume should belong to '+answer+' department')

if __name__ == '__main__':

    app.run()
    
    
