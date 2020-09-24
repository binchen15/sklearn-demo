import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

# read in all columns
data = pd.read_csv('data/students/student-mat.csv', sep=";")
print(data.head())

# extract columns needed.
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
print(data.head())

Predict = "G3"

X = np.array(data.drop([Predict], 1))
y = np.array(data[Predict])


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1)  # this scramble the data

linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
accuracy = linear.score(X_test, y_test)
print(accuracy)

with open("student_math_model.pickle", "wb") as f:
    pickle.dump(linear, f)

pickle_in = open("student_math_model.pickle", "rb")
linear = pickle.load(pickle_in)


print("Coefficients: {}, intercept/bias {}".format(linear.coef_, linear.intercept_))

predictions = linear.predict(X_test)

for i in range(len(predictions)):
    print("{} : {} vs {:.2f}".format(X_test[i], y_test[i], predictions[i]))

style.use('ggplot')
plt.scatter(data['G1'], data['G3'])
plt.xlabel('G1')
plt.ylabel('Final Grade')
plt.show()


