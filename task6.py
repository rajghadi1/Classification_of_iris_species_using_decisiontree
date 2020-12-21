import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv("Iris.csv")
df.drop('Id',inplace=True,axis=1)
x = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
#print(classifier.predict([[7.0,3.2,4.7,1.4]]))


pickle.dump(classifier,open('gripmodel.pkl','wb'))

model=pickle.load(open('gripmodel.pkl','rb'))
#print(model.predict([[5.1,3.5,1.4,0.2]]))








