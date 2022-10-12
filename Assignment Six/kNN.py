# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

filename = 'diabetes.csv'
dataframe = read_csv(filename)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=5, random_state=2, shuffle=True)
model = KNeighborsClassifier(n_neighbors=3)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
