import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
y = cancer.target

classes = ["malignant", "benign"]

X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# print(X_train, y_train)

svc = svm.SVC(kernel="linear")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
