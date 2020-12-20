import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle


df = pd.read_csv("Iris.csv")
df.drop('Id',inplace=True,axis=1)
print(df.head())
x = df.iloc[:, 0:-1].values # Independent variables

y = df.iloc[:, -1].values  # Dependent or Target variable (Iris-Species)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
print(classifier.predict([[7.0,3.2,4.7,1.4]]))


#y_pred = classifier.predict(x_test)
#cm = confusion_matrix(y_test, y_pred)
#print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))


pickle.dump(classifier,open('gripmodel.pkl','wb'))

model=pickle.load(open('gripmodel.pkl','rb'))
print(model.predict([[5.1,3.5,1.4,0.2]]))








