import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df=pd.read_csv('data_moods_features.csv', sep=',',header=None)
X = df[df.columns[5:-2]]
Y = df[df.columns[-1]]
#print(X)
X, Y = np.array(X), np.array(Y)
#print(X)
#print(Y)

X, Y = shuffle(X, Y, random_state=0)
#np.random.shuffle(X)
splitIndex = int(0.75 * len(X))
Xtr, Xva, Ytr, Yva = X[:splitIndex], X[splitIndex:], Y[:splitIndex], Y[splitIndex:]
print(len(Xtr), len(Xva), len(Ytr), len(Yva))

counts = {'Happy' : 0, 'Sad' : 0, 'Energetic' : 0, 'Calm' : 0}
for line in Ytr:
    if 'Happy' in line:
        counts['Happy'] += 1
    elif 'Sad' in line:
        counts['Sad'] += 1
    elif 'Energetic' in line:
        counts['Energetic'] += 1
    else:
        counts['Calm'] += 1
print(counts)

import sklearn.linear_model as lm
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

degrees = [1,2,3,4,5,6]
training_scores, validation_scores = [], []

for degree in degrees:
    clf = lm.LogisticRegression(penalty = 'l2', max_iter = 500)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    X_scaled = scaler.transform(Xtr)

    poly = PolynomialFeatures(degree = degree, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    clf.fit(X_poly, Ytr)
    clf.predict(X_poly)
    training_scores.append(clf.score(X_poly, Ytr))

    Xva_scaled = scaler.transform(Xva)
    Xva_poly = poly.fit_transform(Xva_scaled)
    print(clf.score(Xva_poly, Yva))
    validation_scores.append(clf.score(Xva_poly, Yva))

print(training_scores, validation_scores)

import matplotlib.pyplot as plt

training = plt.plot(degrees, training_scores, label = 'training')
validation = plt.plot(degrees, validation_scores, label = 'validation')
plt.legend(handles=[training[0], validation[0]])
axes = plt.gca()
axes.set_ylim([0,1.1])
plt.show()

tr_err = [1 - x for x in training_scores]
va_err = [1 - x for x in validation_scores]

training = plt.plot(degrees, tr_err, label = 'training')
validation = plt.plot(degrees, va_err, label = 'validation')
plt.legend(handles=[training[0], validation[0]])
plt.show()

from sklearn.tree import DecisionTreeClassifier

depths = [i for i in range(1, 16)]
features = [i for i in range(1, 14)]
scores = []
tmp_scores = []

for depth in depths:
    for feature in features:
        dt = DecisionTreeClassifier(max_features = feature, max_depth = depth, criterion = 'entropy', random_state=0)
        dt.fit(Xtr, Ytr)
        tmp_scores.append(dt.score(Xva, Yva))
    scores.append(tmp_scores)
    tmp_scores = []
max_score = 0
indices = (0, 0)
for index in range(len(scores)):
    for col in scores[index]:
        if col > max_score:
            max_score = col
            indices = (index, scores[index].index(col))
print(max_score, indices)