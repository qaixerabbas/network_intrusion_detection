import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from os import path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import time

bin_data_path = "./datasets/bin_data.csv"
multi_data_path = "./datasets/multi_data.csv"
df = pd.read_csv(bin_data_path)
print("Dimensions of the Training set:", df.shape)
df.shape
df.head()
X = df.drop(columns=["label"], axis=1)
Y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=50
)


knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel="linear", C=1.0, random_state=0)
rf = RandomForestClassifier(n_estimators=10, random_state=1)
dt = DecisionTreeClassifier(random_state=0)
mlp = MLPClassifier(random_state=0, max_iter=300)
clf_voting = VotingClassifier(
    estimators=[("rf", rf), ("knn", knn), ("svm", svm)], voting="hard"
)
knn = KNeighborsClassifier(
    algorithm="auto",
    leaf_size=30,
    metric="minkowski",
    metric_params=None,
    n_jobs=None,
    n_neighbors=5,
    p=2,
    weights="uniform",
)
print("=========================")
print("kNN Classifier")
print("=========================")
t1_ens = time.time()
knn.fit(X_train, y_train.astype(int))
t2_ens = time.time()
print("Time to train knn on training dat:", t2_ens - t1_ens)
y_pred = knn.predict(X_test)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print(cls_report)
pkl_filename = "./qaiser_models/knn_binary.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(knn, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("=========================")
print("Fitting SVM Classifier")
print("=========================")
t1_svm = time.time()
svm.fit(X_train, y_train.astype(int))
t2_svm = time.time()
print("Time to train SVM on training dat:", t2_svm - t1_svm)
y_pred = svm.predict(X_test)
print("Accuracy for binary SVM is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print(cls_report)
pkl_filename = "./qaiser_models/SVM_binary.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(knn, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("=========================")
print("Fitting Random Forest Classifier")
print("=========================")
t1_rf = time.time()
rf.fit(X_train, y_train.astype(int))
t2_rf = time.time()
print("Time to train RF on binary training dat:", t2_rf - t1_rf)
print("======================================================")
y_pred = rf.predict(X_test)
print("Accuracy for binary SVM is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print("========Printing Classification Reports==========")
print(cls_report)
pkl_filename = "./qaiser_models/RF_binary.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(rf, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("===========================================")
print("Fitting Random Forest Classifier")
print("===========================================")
t1_dt = time.time()
dt.fit(X_train, y_train.astype(int))
t2_dt = time.time()
print("Time to train RF on binary training dat:", t2_dt - t1_dt)
print("======================================================")
y_pred = dt.predict(X_test)
print("Accuracy for binary SVM is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print("========Printing Classification Reports==========")
print(cls_report)
pkl_filename = "./qaiser_models/DT_binary.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(dt, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("===========================================")
print("Fitting MLP Classifier")
print("===========================================")
t1_mlp = time.time()
mlp.fit(X_train, y_train.astype(int))
t2_mlp = time.time()
print("Time to train MLP on binary training dat:", t2_dt - t1_dt)
print("======================================================")
y_pred = mlp.predict(X_test)
print("Accuracy for binary MLP is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print("========Printing Classification Reports==========")
print(cls_report)
pkl_filename = "./qaiser_models/MLP_binary.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(mlp, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("===========================================")
print("Fitting Our Ensemble Method Classifier")
print("===========================================")
t1_clf_voting = time.time()
clf_voting.fit(X_train, y_train.astype(int))
t2_clf_voting = time.time()
print("Time to train clf_voting on binary training dat:", t2_clf_voting - t1_clf_voting)
print("======================================================")
y_pred = clf_voting.predict(X_test)
print("Accuracy for binary clf_voting is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print("========Printing Classification Reports==========")
print(cls_report)
pkl_filename = "./qaiser_models/clf_voting_binary.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(clf_voting, file)
    print("Saved model to disk")
else:
    print("Model already saved")
