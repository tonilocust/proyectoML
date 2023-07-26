import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('train.csv')

X = df.drop(columns=['red', 'white', 'quality'], axis=1)
y = df['red']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

norm = MinMaxScaler()

norm_fit = norm.fit(X_train)
X_train_norm = norm_fit.transform(X_train)
X_test_norm = norm_fit.transform(X_test)

rf = RandomForestClassifier(n_estimators=300,
                            min_samples_split=2,
                            random_state=42)
rf.fit(X_train_norm, y_train)

y_pred = rf.predict(X_test_norm)

accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(rf, X_train_norm, y_train, cv=5, scoring='accuracy')

print("Cross-validation scores (train):", cv_scores)

print("Mean Accuracy (train):", cv_scores.mean())
print("Standard Deviation (train):", cv_scores.std())


df2 = pd.read_csv('test.csv')

X = df2.drop(columns=['red', 'white', 'quality'], axis=1)
y = df2['red']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=300,
                            min_samples_split=2,
                            random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_test, y_pred)


import pickle

with open('new_model.pkl', 'wb') as f:
    pickle.dump(rf, f)