from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from csv import writer
import pandas as pd
from flask_material import Material
from feature import generate_data_set
from sklearn.ensemble import GradientBoostingClassifier
UPLOAD_FOLDER = 'static/uploads/'


data = pd.read_csv("phishing.csv")
#droping index column
data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

clf_rfecv_model = gbc.fit(X_train, y_train)
gbc.fit(X_train, y_train)


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'




# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/')
def home():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/domain')
def domain():
    return render_template('domain.html')


@app.route('/hosting')
def hosting():
    return render_template('hosting.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30) 
        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('domain.html',xx =round(y_pro_non_phishing,2),url=url )
        # else:
        #     pred = "It is {0:.2f} % unsafe to go ".format(y_pro_non_phishing*100)
        #     return render_template('index.html',x =y_pro_non_phishing,url=url )
    return render_template("domain.html", xx =-1)

    
if __name__ =='__main__':
	app.run()
