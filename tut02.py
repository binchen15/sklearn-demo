import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('./data/car/car.data')
print(data.head())

# convert labels into numbers
encoder = preprocessing.LabelEncoder()
buying = encoder.fit_transform(list(data['buying']))
maint = encoder.fit_transform(list(data['maint']))
door = encoder.fit_transform(list(data['door']))
persons = encoder.fit_transform(list(data['persons']))
lug_boot = encoder.fit_transform(list(data['lug_boot']))
safety = encoder.fit_transform(list(data['safety']))
cls = encoder.fit_transform(list(data['class']))

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1)

#print(X_test, y_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train) # all data must be saved.
accuracy = model.score(X_test, y_test)
print(accuracy)

predicted = model.predict(X_test)
names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(predicted)):
    print("Predicted: {} vs {} actual"
          .format(predicted[x], y_test[x]))
    # n = model.kneighbors([X_test[x]], 9, True)
    # print(n)

