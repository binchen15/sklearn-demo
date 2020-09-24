import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)
y = digits.target

k = 10  # assuming 10 digits

samples, features = data.shape  # (1000, 728)

clf = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300)
clf.fit(data)
print(clf.labels_)
# print(y)
