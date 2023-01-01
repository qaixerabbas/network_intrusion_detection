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

bin_data_path = "./datasets/multi_data.csv"
df = pd.read_csv(bin_data_path)
print("Dimensions of the Training set:", df.shape)
df.shape
df.head()
X = df.drop(columns=["label"], axis=1)
Y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=50
)
print("Training Data Shape:", X_train.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Testing Label Shape:", y_test.shape)

knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel="poly", C=1.0, random_state=0)
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
print("Time to train knn on MultiClass training dat:", t2_ens - t1_ens)
y_pred = knn.predict(X_test)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print(cls_report)
pkl_filename = "./qaiser_models/knn_multi.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(knn, file)
    print("Saved model to disk")
else:
    print("Model already saved")
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

print("Testing on Unssen Data")
fig, ax = plt.subplots(figsize=(10, 10))
labels = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Recon",
    "Worms",
]
plot_confusion_matrix(
    knn, X_test, y_test, cmap="Greens", display_labels=labels, normalize="pred", ax=ax
)
plt.savefig("./diagrams/kNN Confusion Matrix.png")
plt.show()
print("=========================")
print("Fitting SVM Classifier")
print("=========================")
t1_svm = time.time()
svm.fit(X_train, y_train.astype(int))
t2_svm = time.time()
print("Time to train SVM on training dat:", t2_svm - t1_svm)
y_pred = svm.predict(X_test)
print("Accuracy for Multiclass SVM is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print(cls_report)
pkl_filename = "./qaiser_models/SVM_multi.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(svm, file)
    print("Saved model to disk")
else:
    print("Model already saved")
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

print("Testing on Unssen Data")
fig, ax = plt.subplots(figsize=(10, 10))
labels = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Recon",
    "Worms",
]
plot_confusion_matrix(
    svm, X_test, y_test, cmap="Greens", display_labels=labels, normalize="pred", ax=ax
)
plt.savefig("./diagrams/SVM multiclass Confusion Matrix.png")
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

print("=========================")
print("Fitting Random Forest Classifier")
print("=========================")
t1_rf = time.time()
rf.fit(X_train, y_train.astype(int))
t2_rf = time.time()
print("Time to train RF on binary training dat:", t2_rf - t1_rf)
print("======================================================")
y_pred = rf.predict(X_test)
print("Accuracy for multi RF is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print("========Printing Classification Reports==========")
print(cls_report)
pkl_filename = "./qaiser_models/RF_multi.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(rf, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("Testing on Unssen Data")
fig, ax = plt.subplots(figsize=(10, 10))
labels = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Recon",
    "Worms",
]
plot_confusion_matrix(
    rf, X_test, y_test, cmap="Greens", display_labels=labels, normalize="pred", ax=ax
)
plt.savefig("./diagrams/RF Confusion Matrix.png")
plt.show()
print("===========================================")
print("Fitting DT Classifier")
print("===========================================")
t1_dt = time.time()
dt.fit(X_train, y_train.astype(int))
t2_dt = time.time()
print("Time to train RF on multiclass training dat:", t2_dt - t1_dt)
print("======================================================")
y_pred = dt.predict(X_test)
print("Accuracy for multi DT is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print("========Printing Classification Reports==========")
print(cls_report)
pkl_filename = "./qaiser_models/DT_multi.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(dt, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("Testing on Unssen Data")
fig, ax = plt.subplots(figsize=(10, 10))
labels = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Recon",
    "Worms",
]
plot_confusion_matrix(
    dt, X_test, y_test, cmap="Greens", display_labels=labels, normalize="pred", ax=ax
)
plt.savefig("./diagrams/DT Confusion Matrix.png")
plt.show()
print("===========================================")
print("Fitting MLP Classifier")
print("===========================================")
t1_mlp = time.time()
mlp.fit(X_train, y_train.astype(int))
t2_mlp = time.time()
print("Time to train MLP on multiclass training dat:", t2_dt - t1_dt)
print("======================================================")
y_pred = mlp.predict(X_test)
print("Accuracy for multiclass MLP is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print("========Printing Classification Reports==========")
print(cls_report)
pkl_filename = "./qaiser_models/MLP_multi.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(mlp, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("Testing on Unssen Data")
fig, ax = plt.subplots(figsize=(10, 10))
labels = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Recon",
    "Worms",
]
plot_confusion_matrix(
    mlp, X_test, y_test, cmap="Greens", display_labels=labels, normalize="pred", ax=ax
)
plt.savefig("./diagrams/MLP Confusion Matrix.png")
plt.show()
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
pkl_filename = "./qaiser_models/clf_voting_multi.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(clf_voting, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("Testing on Unssen Data")
fig, ax = plt.subplots(figsize=(10, 10))
labels = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Recon",
    "Worms",
]
plot_confusion_matrix(
    clf_voting,
    X_test,
    y_test,
    cmap="Greens",
    display_labels=labels,
    normalize="pred",
    ax=ax,
)
plt.savefig("./diagrams/clf_voting Confusion Matrix-Testing.png")
plt.show()
print("===========================================")
print("Fitting Our Ensemble Method Classifier")
print("===========================================")
from sklearn.ensemble import GradientBoostingClassifier

xg = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0
)
clf_voting = VotingClassifier(
    estimators=[("rf", rf), ("dt", dt), ("xg", xg)], voting="hard"
)
t1_clf_voting = time.time()
clf_voting.fit(X_train, y_train.astype(int))
t2_clf_voting = time.time()
print("Time to train clf_voting on multi training dat:", t2_clf_voting - t1_clf_voting)
print("======================================================")
y_pred = clf_voting.predict(X_test)
print("Accuracy for multiclass clf_voting is - ", accuracy_score(y_test, y_pred) * 100)
cls_report = classification_report(y_true=y_test, y_pred=y_pred)
print("========Printing Classification Reports==========")
print(cls_report)
pkl_filename = "./qaiser_models/clf_ensemble_multi.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, "wb") as file:
        pickle.dump(clf_voting, file)
    print("Saved model to disk")
else:
    print("Model already saved")
print("Testing on Unssen Data")
fig, ax = plt.subplots(figsize=(10, 10))
labels = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Recon",
    "Worms",
]
plot_confusion_matrix(
    clf_voting,
    X_test,
    y_test,
    cmap="Greens",
    display_labels=labels,
    normalize="pred",
    ax=ax,
)
plt.savefig("./diagrams/Ensemble Confusion Matrix-Testing.png")
plt.show()
print("===========================================")
print("Fitting Our Ensemble Method Classifier")
print("===========================================")
print("Time to train clf_voting on multi training dat:", t2_clf_voting - t1_clf_voting)
print("======================================================")
y_pred = clf_voting.predict(X_train)
print(
    "Accuracy for multiclass clf_voting on Training Data is - ",
    accuracy_score(y_train.astype(int), y_pred) * 100,
)
print("Testing on Unssen Data")
fig, ax = plt.subplots(figsize=(10, 10))
labels = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Recon",
    "Worms",
]
plot_confusion_matrix(
    clf_voting,
    X_train,
    y_train,
    cmap="Greens",
    display_labels=labels,
    normalize="pred",
    ax=ax,
)
plt.savefig("./diagrams/Ensemble Training Data Confusion Matrix-Testing.png")
plt.show()
xg = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0
)
clf_voting = VotingClassifier(
    estimators=[("rf", rf), ("dt", dt), ("xg", xg)], voting="soft"
)
t1_clf_voting = time.time()
clf_voting.fit(X_train, y_train.astype(int))
t2_clf_voting = time.time()
print("Time to train clf_voting on multi training dat:", t2_clf_voting - t1_clf_voting)
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt

print("ROC Curve for Testing Data")
preds = clf_voting.predict_proba(X_test)
fig, ax = plt.subplots(figsize=(10, 10))
skplt.plot_roc_curve(y_test, preds, ax=ax)
plt.savefig("Ensemble ROC for Testing.png")
plt.show()
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt

print("ROC Curve for Training Data")
preds = clf_voting.predict_proba(X_train)
fig, ax = plt.subplots(figsize=(10, 10))
skplt.plot_roc_curve(y_train, preds, ax=ax)
plt.savefig("Ensemble ROC for Training.png")
plt.show()
