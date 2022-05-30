import pickle
from audioop import reverse

import os
from flask import Flask, render_template, request, redirect
import numpy as np
from joblib import load
from flask import Flask
import pandas as pd
import csv, json
from flask_restful import Api
from pyexpat.errors import messages
from resources.churn import ChurnAPI
import config
from databases.model import db,Df
import sys
import glob
import joblib
import pickle

app = Flask(__name__)
app.config.from_object(config)

db.init_app(app)
api = Api(app)



## Declaration of API routes

api.add_resource(ChurnAPI, '/churn')

filepath = ""


@app.route('/users', methods=["GET", "POST"])
def users():
    data = []
    if request.method == 'POST':
        if request.files:
            uploaded_file = request.files['filename']  # This line uses the same variable and worked fine
            global filepath
            filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
            uploaded_file.save(filepath)
            with open(filepath) as file:
                csv_file = csv.reader(file)
                for row in csv_file:
                    data.append(row)
        return redirect('/predict')
    return render_template('users.html')

@app.route('/admin', methods=["GET", "POST"])
def admin():
    data = []
    if request.method == 'POST':
        if request.files:
            uploaded_file = request.files['filename'] # This line uses the same variable and worked fine
            global filepath
            filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
            uploaded_file.save(filepath)
            with open(filepath) as file:
                csv_file = csv.reader(file)
                for row in csv_file:
                    data.append(row)
        return redirect('/relearn')
    return render_template('admin.html')

@app.route('/', methods=["GET", "POST"])
def index():
    # data = []
    # if request.method == 'POST':
    #     if request.files:
    #         uploaded_file = request.files['filename'] # This line uses the same variable and worked fine
    #         global filepath
    #         filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
    #         uploaded_file.save(filepath)
    #         with open(filepath) as file:
    #             csv_file = csv.reader(file)
    #             for row in csv_file:
    #                 data.append(row)
    #         return redirect('/predict')
   # return render_template('index.html', data=data)
    return render_template('index.html')
@app.route("/relearn")
def relearn():
    global filepath
    df = pd.read_csv(filepath)

    df = pd.concat([df, pd.get_dummies(df['Gender'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Geography'])], axis=1)
    Y = df['Exited']
    X = df.drop(columns=['Exited', 'Geography', 'Gender', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
    X = df[
        ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
         'Female', 'Male', 'France', 'Germany', 'Spain']]

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=69420)

    from imblearn.over_sampling import SMOTE
    sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=100)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)  # uncomment for oversmapling

    S = MinMaxScaler()
    S.fit(X_train)
    X_train = S.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X_test.columns.tolist())
    X_train.columns = X.columns
    X_test = S.transform(X_test)
    X = S.transform(X)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, \
        f1_score

    print('##################################')
    print('###   RandomForestClassifier   ###')
    print('##################################')
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    filepath2 = os.path.join(app.config['FILE_MODELS'], 'Churn_normalizer.pkl')
    joblib.dump(S, filepath2)
    filepath2 = os.path.join(app.config['FILE_MODELS'], 'Churn_model.pkl')
    joblib.dump(model, filepath2)


    return redirect('/accuracy')


@app.route("/accuracy")
def accuracy():
    S = joblib.load('./models/Churn_normalizer.pkl')
    model = joblib.load('./models/Churn_model.pkl')
    global filepath
    df = pd.read_csv(filepath)

    df = pd.concat([df, pd.get_dummies(df['Gender'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Geography'])], axis=1)
    Y = df['Exited']
    X = df.drop(columns=['Exited', 'Geography', 'Gender', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
    X = df[
        ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
         'Female', 'Male', 'France', 'Germany', 'Spain']]
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=69420)
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
    #text1 = 'Accuracy: {:.0%}'.format(accuracy_score(Y_test, Y_predicted)), 'Recall: {:.0%}'.format(recall_score(Y_test, Y_predicted)), 'Precision: {:.0%}'.format(precision_score(Y_test, Y_predicted)), 'F1 score: {:.0%}'.format(f1_score(Y_test, Y_predicted)), 'Confusion matrix:{}'.format(confusion_matrix(Y_test, Y_predicted))
    text1 = 'Accuracy: {:.0%}'.format(accuracy_score(Y_test, Y_predicted)), 'Recall: {:.0%}'.format(recall_score(Y_test, Y_predicted)), 'Precision: {:.0%}'.format(precision_score(Y_test, Y_predicted)), 'F1 score: {:.0%}'.format(f1_score(Y_test, Y_predicted))

    return render_template("accuracy.html", prediction_text=text1)




@app.route("/predict")
def hello():
    global filepath
    df = pd.read_csv(filepath)
    df = pd.concat([df, pd.get_dummies(df['Gender'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Geography'])], axis=1)

    filepath = os.path.join(app.config['FILE_MODELS'], 'Churn_normalizer.pkl')
    S = joblib.load(filepath)
    filepath = os.path.join(app.config['FILE_MODELS'], 'Churn_model.pkl')
    model = joblib.load(filepath)

    X = df.drop(columns= ['Exited','Geography','Gender','RowNumber','CustomerId','Surname'], axis=1)
    X = df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Female','Male','France','Germany','Spain']]
    X  = S.transform(X)

    df['chance of leaving'] = model.predict_proba (X)[:,1]
    df['chance of leaving'] = df['chance of leaving'] * 100

    df = df.sort_values('chance of leaving', ascending=False)


    return df[['Surname', 'chance of leaving']].head(100).to_html()

    #df.nlargest (500, 'mayleave')[['Surname', 'change_leaving_percentage']]


'''
    Y = df['Exited']
    X = df.drop(columns=['Exited', 'Geography', 'Gender', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
    X = df[
        ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
         'Female', 'Male', 'France', 'Germany', 'Spain']]

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=69420)

    from imblearn.over_sampling import SMOTE
    sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=100)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)  # uncomment for oversmapling

    S = MinMaxScaler()
    S.fit(X_train)
    X_train = S.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X_test.columns.tolist())
    X_train.columns = X.columns
    X_test = S.transform(X_test)
    X = S.transform(X)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, \
        f1_score

    print('##################################')
    print('###   RandomForestClassifier   ###')
    print('##################################')
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    np.set_printoptions(threshold=sys.maxsize)
    # np.set_printoptions(threshold=1000)

    df['mayleave'] = model.predict_proba(X)[:, 1]
    df['mayleave'] = df['mayleave'] * 100
    # print (model.predict(X))
    # df
    df = df.sort_values('mayleave', ascending=False)


    return df[['Surname', 'mayleave']].head(100).to_html()
'''

class HttpResponseRedirect:
    pass


app.config['FILE_UPLOADS'] = "..\\app\\databases"
app.config['FILE_MODELS'] = "..\\app\\models"

if __name__ == "__main__":
    app.run(debug=True)