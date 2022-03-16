import pandas as pd
from pandas.core.algorithms import mode
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("titanic.csv")

train.dropna(inplace=True)

target = 'Survived'
features = ['Pclass',"Age",'SibSp','Fare']

x = train[features]
y = train[target]

model =  LogisticRegression()
model.fit(x,y)
model.score(x,y)

import pickle
pickle.dump(model,open("model.pkl",'wb'))