print(__doc__)

import matplotlib.pyplot as plt


from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r = lda.fit(X_train, y_train).transform(X_train)


# Percentage of variance explained for each components

#plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y_train == i, 0], X_r[y_train == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
lda.transform(X_test)
predictions = lda.predict(X_test)
print('Accuracy')
print(accuracy_score(predictions,y_test))
print('Matriz confusao')
print(confusion_matrix(y_test, predictions))
print('Precisao')
print(classification_report(y_test, predictions))
print('MCC')
print(matthews_corrcoef(y_test, predictions))
print('Recall')
print(recall_score(y_test, predictions,average = 'macro'))


plt.show()
