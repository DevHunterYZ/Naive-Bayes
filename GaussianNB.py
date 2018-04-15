# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# İris verisetini yükle.
dataset = datasets.load_iris()
# Verileri bir Naive Bayes modeliyle eğit.
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)
# Tahmin yap.
expected = dataset.target
predicted = model.predict(dataset.data)
# Modelin başarısını ölç.
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
