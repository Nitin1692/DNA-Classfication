from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
cv = CountVectorizer(ngram_range=(4,4))

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    txtfile = request.files['txtfile']
    txtfile_path = "./text/" + txtfile.filename
    txtfile.save(txtfile_path)
    human_data = pd.read_table(txtfile_path)
    human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
    human_data = human_data.drop('sequence', axis=1)
    human_texts = list(human_data["words"]) 
    for item in range (len(human_data)):
        human_texts[item]=" ".join(human_texts[item])
    X_human = cv.fit_transform(human_texts) 
    y_human = human_data.iloc[:, 0].values
    X_train_human, X_test_human, y_train_human, y_test_human = train_test_split(X_human,y_human,test_size = 0.25,random_state=42)
    dtree_human = DecisionTreeClassifier() 
    dtree_human.fit(X_train_human,y_train_human)
    dtree_human_pred = dtree_human.predict(X_test_human)
    pred = accuracy_score(dtree_human_pred,y_test_human) 
    return render_template('index.html', prediction=pred)

@app.route('/',methods=['POST'])
def predict_random():
    txtfile = request.files['txtfile']
    txtfile_path = "./text/" + txtfile.filename
    txtfile.save(txtfile_path)
    human_data = pd.read_table(txtfile_path)
    human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
    human_data = human_data.drop('sequence', axis=1)
    human_texts = list(human_data["words"]) 
    for item in range (len(human_data)):
        human_texts[item]=" ".join(human_texts[item])
    X_human = cv.fit_transform(human_texts) 
    y_human = human_data.iloc[:, 0].values
    X_train_human, X_test_human, y_train_human, y_test_human = train_test_split(X_human,y_human,test_size = 0.25,random_state=42)
    rf_human = RandomForestClassifier(n_estimators=400)
    rf_human.fit(X_train_human,y_train_human)
    rf_human_pred = rf_human.predict(X_test_human)
    pred_random = accuracy_score(rf_human_pred,y_test_human)
    return render_template('index.html', prediction=pred_random)

@app.route('/',methods=['POST'])
def predict_xgboost():
    txtfile = request.files['txtfile']
    txtfile_path = "./text/" + txtfile.filename
    txtfile.save(txtfile_path)
    human_data = pd.read_table(txtfile_path)
    human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
    human_data = human_data.drop('sequence', axis=1)
    human_texts = list(human_data["words"]) 
    for item in range (len(human_data)):
        human_texts[item]=" ".join(human_texts[item])
    X_human = cv.fit_transform(human_texts) 
    y_human = human_data.iloc[:, 0].values
    X_train_human, X_test_human, y_train_human, y_test_human = train_test_split(X_human,y_human,test_size = 0.25,random_state=42)
    xgb_human = xgb.XGBClassifier()  
    xgb_human.fit(X_train_human,y_train_human) 
    xgb_human_pred = xgb_human.predict(X_test_human)  
    pred_xgboost = accuracy_score(xgb_human_pred,y_test_human)
    return render_template('index.html', prediction=pred_xgboost)


@app.route('/',methods=['POST'])
def predict_naive():
    txtfile = request.files['txtfile']
    txtfile_path = "./text/" + txtfile.filename
    txtfile.save(txtfile_path)
    human_data = pd.read_table(txtfile_path)
    human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
    human_data = human_data.drop('sequence', axis=1)
    human_texts = list(human_data["words"]) 
    for item in range (len(human_data)):
        human_texts[item]=" ".join(human_texts[item])
    X_human = cv.fit_transform(human_texts) 
    y_human = human_data.iloc[:, 0].values
    X_train_human, X_test_human, y_train_human, y_test_human = train_test_split(X_human,y_human,test_size = 0.25,random_state=42)
    NB_human = MultinomialNB(alpha=0.1) 
    NB_human.fit(X_train_human,y_train_human) 
    y_pred_human = NB_human.predict(X_test_human)  
    nb_accuracy_human = accuracy_score(y_pred_human,y_test_human)
    return render_template('index.html', prediction=nb_accuracy_human)



if __name__ == '__main__':
    app.run(port=3000,debug=True)