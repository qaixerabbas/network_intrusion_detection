import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import sys
import sklearn
import io
import random

train_url = "https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Train.csv"
test_url = "https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv"
col_names = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]
df = pd.read_csv(train_url, header=None, names=col_names)
df_test = pd.read_csv(test_url, header=None, names=col_names)
print("Dimensions of the Training set:", df.shape)
print("Dimensions of the Test set:", df_test.shape)
df.head(5)
print("Label distribution Training set:")
print(df["label"].value_counts())
print()
print("Label distribution Test set:")
print(df_test["label"].value_counts())
print("Training set:")
for col_name in df.columns:
    if df[col_name].dtypes == "object":
        unique_cat = len(df[col_name].unique())
        print(
            "Feature '{col_name}' has {unique_cat} categories".format(
                col_name=col_name, unique_cat=unique_cat
            )
        )
print()
print("Distribution of categories in service:")
print(df["service"].value_counts().sort_values(ascending=False).head())
print("Test set:")
for col_name in df_test.columns:
    if df_test[col_name].dtypes == "object":
        unique_cat = len(df_test[col_name].unique())
        print(
            "Feature '{col_name}' has {unique_cat} categories".format(
                col_name=col_name, unique_cat=unique_cat
            )
        )
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

categorical_columns = ["protocol_type", "service", "flag"]
df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
df_categorical_values.head()
unique_protocol = sorted(df.protocol_type.unique())
string1 = "Protocol_type_"
unique_protocol2 = [string1 + x for x in unique_protocol]
print(unique_protocol2)
unique_service = sorted(df.service.unique())
string2 = "service_"
unique_service2 = [string2 + x for x in unique_service]
print(unique_service2)
unique_flag = sorted(df.flag.unique())
string3 = "flag_"
unique_flag2 = [string3 + x for x in unique_flag]
print(unique_flag2)
dumcols = unique_protocol2 + unique_service2 + unique_flag2
unique_service_test = sorted(df_test.service.unique())
unique_service2_test = [string2 + x for x in unique_service_test]
testdumcols = unique_protocol2 + unique_service2_test + unique_flag2
df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)
print(df_categorical_values.head())
print("--------------------")
print(df_categorical_values_enc.head())
testdf_categorical_values_enc = testdf_categorical_values.apply(
    LabelEncoder().fit_transform
)
enc = OneHotEncoder(categories="auto")
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(), columns=dumcols)
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(
    testdf_categorical_values_encenc.toarray(), columns=testdumcols
)
df_cat_data.head()
trainservice = df["service"].tolist()
testservice = df_test["service"].tolist()
difference = list(set(trainservice) - set(testservice))
string = "service_"
difference = [string + x for x in difference]
difference
for col in difference:
    testdf_cat_data[col] = 0
print(df_cat_data.shape)
print(testdf_cat_data.shape)
newdf = df.join(df_cat_data)
newdf.drop("flag", axis=1, inplace=True)
newdf.drop("protocol_type", axis=1, inplace=True)
newdf.drop("service", axis=1, inplace=True)
newdf_test = df_test.join(testdf_cat_data)
newdf_test.drop("flag", axis=1, inplace=True)
newdf_test.drop("protocol_type", axis=1, inplace=True)
newdf_test.drop("service", axis=1, inplace=True)
print(newdf.shape)
print(newdf_test.shape)
labeldf = newdf["label"]
labeldf_test = newdf_test["label"]
newlabeldf = labeldf.replace(
    {
        "normal": 0,
        "neptune": 1,
        "back": 1,
        "land": 1,
        "pod": 1,
        "smurf": 1,
        "teardrop": 1,
        "mailbomb": 1,
        "apache2": 1,
        "processtable": 1,
        "udpstorm": 1,
        "worm": 1,
        "ipsweep": 2,
        "nmap": 2,
        "portsweep": 2,
        "satan": 2,
        "mscan": 2,
        "saint": 2,
        "ftp_write": 3,
        "guess_passwd": 3,
        "imap": 3,
        "multihop": 3,
        "phf": 3,
        "spy": 3,
        "warezclient": 3,
        "warezmaster": 3,
        "sendmail": 3,
        "named": 3,
        "snmpgetattack": 3,
        "snmpguess": 3,
        "xlock": 3,
        "xsnoop": 3,
        "httptunnel": 3,
        "buffer_overflow": 4,
        "loadmodule": 4,
        "perl": 4,
        "rootkit": 4,
        "ps": 4,
        "sqlattack": 4,
        "xterm": 4,
    }
)
newlabeldf_test = labeldf_test.replace(
    {
        "normal": 0,
        "neptune": 1,
        "back": 1,
        "land": 1,
        "pod": 1,
        "smurf": 1,
        "teardrop": 1,
        "mailbomb": 1,
        "apache2": 1,
        "processtable": 1,
        "udpstorm": 1,
        "worm": 1,
        "ipsweep": 2,
        "nmap": 2,
        "portsweep": 2,
        "satan": 2,
        "mscan": 2,
        "saint": 2,
        "ftp_write": 3,
        "guess_passwd": 3,
        "imap": 3,
        "multihop": 3,
        "phf": 3,
        "spy": 3,
        "warezclient": 3,
        "warezmaster": 3,
        "sendmail": 3,
        "named": 3,
        "snmpgetattack": 3,
        "snmpguess": 3,
        "xlock": 3,
        "xsnoop": 3,
        "httptunnel": 3,
        "buffer_overflow": 4,
        "loadmodule": 4,
        "perl": 4,
        "rootkit": 4,
        "ps": 4,
        "sqlattack": 4,
        "xterm": 4,
    }
)
newdf["label"] = newlabeldf
newdf_test["label"] = newlabeldf_test
to_drop_DoS = [0, 1]
to_drop_Probe = [0, 2]
to_drop_R2L = [0, 3]
to_drop_U2R = [0, 4]
DoS_df = newdf[newdf["label"].isin(to_drop_DoS)]
Probe_df = newdf[newdf["label"].isin(to_drop_Probe)]
R2L_df = newdf[newdf["label"].isin(to_drop_R2L)]
U2R_df = newdf[newdf["label"].isin(to_drop_U2R)]
DoS_df_test = newdf_test[newdf_test["label"].isin(to_drop_DoS)]
Probe_df_test = newdf_test[newdf_test["label"].isin(to_drop_Probe)]
R2L_df_test = newdf_test[newdf_test["label"].isin(to_drop_R2L)]
U2R_df_test = newdf_test[newdf_test["label"].isin(to_drop_U2R)]
print("Train:")
print("Dimensions of DoS:", DoS_df.shape)
print("Dimensions of Probe:", Probe_df.shape)
print("Dimensions of R2L:", R2L_df.shape)
print("Dimensions of U2R:", U2R_df.shape)
print()
print("Test:")
print("Dimensions of DoS:", DoS_df_test.shape)
print("Dimensions of Probe:", Probe_df_test.shape)
print("Dimensions of R2L:", R2L_df_test.shape)
print("Dimensions of U2R:", U2R_df_test.shape)
X_DoS = DoS_df.drop("label", 1)
Y_DoS = DoS_df.label
X_Probe = Probe_df.drop("label", 1)
Y_Probe = Probe_df.label
X_R2L = R2L_df.drop("label", 1)
Y_R2L = R2L_df.label
X_U2R = U2R_df.drop("label", 1)
Y_U2R = U2R_df.label
X_DoS_test = DoS_df_test.drop("label", 1)
Y_DoS_test = DoS_df_test.label
X_Probe_test = Probe_df_test.drop("label", 1)
Y_Probe_test = Probe_df_test.label
X_R2L_test = R2L_df_test.drop("label", 1)
Y_R2L_test = R2L_df_test.label
X_U2R_test = U2R_df_test.drop("label", 1)
Y_U2R_test = U2R_df_test.label
colNames = list(X_DoS)
colNames_test = list(X_DoS_test)
from sklearn import preprocessing

scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS = scaler1.transform(X_DoS)
scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe = scaler2.transform(X_Probe)
scaler3 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L = scaler3.transform(X_R2L)
scaler4 = preprocessing.StandardScaler().fit(X_U2R)
X_U2R = scaler4.transform(X_U2R)
scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test = scaler5.transform(X_DoS_test)
scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test = scaler6.transform(X_Probe_test)
scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test = scaler7.transform(X_R2L_test)
scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test = scaler8.transform(X_U2R_test)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, n_jobs=2)
rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
rfe.fit(X_DoS, Y_DoS.astype(int))
X_rfeDoS = rfe.transform(X_DoS)
true = rfe.support_
rfecolindex_DoS = [i for i, x in enumerate(true) if x]
rfecolname_DoS = list(colNames[i] for i in rfecolindex_DoS)
rfe.fit(X_Probe, Y_Probe.astype(int))
X_rfeProbe = rfe.transform(X_Probe)
true = rfe.support_
rfecolindex_Probe = [i for i, x in enumerate(true) if x]
rfecolname_Probe = list(colNames[i] for i in rfecolindex_Probe)
rfe.fit(X_R2L, Y_R2L.astype(int))
X_rfeR2L = rfe.transform(X_R2L)
true = rfe.support_
rfecolindex_R2L = [i for i, x in enumerate(true) if x]
rfecolname_R2L = list(colNames[i] for i in rfecolindex_R2L)
rfe.fit(X_U2R, Y_U2R.astype(int))
X_rfeU2R = rfe.transform(X_U2R)
true = rfe.support_
rfecolindex_U2R = [i for i, x in enumerate(true) if x]
rfecolname_U2R = list(colNames[i] for i in rfecolindex_U2R)
print("Features selected for DoS:", rfecolname_DoS)
print()
print("Features selected for Probe:", rfecolname_Probe)
print()
print("Features selected for R2L:", rfecolname_R2L)
print()
print("Features selected for U2R:", rfecolname_U2R)
print(X_rfeDoS.shape)
print(X_rfeProbe.shape)
print(X_rfeR2L.shape)
print(X_rfeU2R.shape)
clf_DoS = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_Probe = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_R2L = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_U2R = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_DoS.fit(X_DoS, Y_DoS.astype(int))
clf_Probe.fit(X_Probe, Y_Probe.astype(int))
clf_R2L.fit(X_R2L, Y_R2L.astype(int))
clf_U2R.fit(X_U2R, Y_U2R.astype(int))
clf_rfeDoS = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_rfeProbe = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_rfeR2L = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_rfeU2R = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_rfeDoS.fit(X_rfeDoS, Y_DoS.astype(int))
clf_rfeProbe.fit(X_rfeProbe, Y_Probe.astype(int))
clf_rfeR2L.fit(X_rfeR2L, Y_R2L.astype(int))
clf_rfeU2R.fit(X_rfeU2R, Y_U2R.astype(int))
clf_DoS.predict(X_DoS_test)
clf_DoS.predict_proba(X_DoS_test)[0:10]
Y_DoS_pred = clf_DoS.predict(X_DoS_test)
pd.crosstab(
    Y_DoS_test, Y_DoS_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_Probe_pred = clf_Probe.predict(X_Probe_test)
pd.crosstab(
    Y_Probe_test,
    Y_Probe_pred,
    rownames=["Actual attacks"],
    colnames=["Predicted attacks"],
)
Y_R2L_pred = clf_R2L.predict(X_R2L_test)
pd.crosstab(
    Y_R2L_test, Y_R2L_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_U2R_pred = clf_U2R.predict(X_U2R_test)
pd.crosstab(
    Y_U2R_test, Y_U2R_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
from sklearn.model_selection import cross_val_score
from sklearn import metrics

accuracy = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="accuracy")
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="precision")
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="recall")
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="f1")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="accuracy")
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="recall_macro")
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="accuracy")
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="recall_macro")
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
X_DoS_test2 = X_DoS_test[:, rfecolindex_DoS]
X_Probe_test2 = X_Probe_test[:, rfecolindex_Probe]
X_R2L_test2 = X_R2L_test[:, rfecolindex_R2L]
X_U2R_test2 = X_U2R_test[:, rfecolindex_U2R]
X_U2R_test2.shape
Y_DoS_pred2 = clf_rfeDoS.predict(X_DoS_test2)
pd.crosstab(
    Y_DoS_test, Y_DoS_pred2, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_Probe_pred2 = clf_rfeProbe.predict(X_Probe_test2)
pd.crosstab(
    Y_Probe_test,
    Y_Probe_pred2,
    rownames=["Actual attacks"],
    colnames=["Predicted attacks"],
)
Y_R2L_pred2 = clf_rfeR2L.predict(X_R2L_test2)
pd.crosstab(
    Y_R2L_test, Y_R2L_pred2, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_U2R_pred2 = clf_rfeU2R.predict(X_U2R_test2)
pd.crosstab(
    Y_U2R_test, Y_U2R_pred2, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
accuracy = cross_val_score(
    clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring="precision"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring="recall")
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring="f1")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(
    clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring="f1_macro"
)
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
from sklearn.neighbors import KNeighborsClassifier

clf_KNN_DoS = KNeighborsClassifier()
clf_KNN_Probe = KNeighborsClassifier()
clf_KNN_R2L = KNeighborsClassifier()
clf_KNN_U2R = KNeighborsClassifier()
clf_KNN_DoS.fit(X_DoS, Y_DoS.astype(int))
clf_KNN_Probe.fit(X_Probe, Y_Probe.astype(int))
clf_KNN_R2L.fit(X_R2L, Y_R2L.astype(int))
clf_KNN_U2R.fit(X_U2R, Y_U2R.astype(int))
Y_DoS_pred = clf_KNN_DoS.predict(X_DoS_test)
pd.crosstab(
    Y_DoS_test, Y_DoS_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_Probe_pred = clf_KNN_Probe.predict(X_Probe_test)
pd.crosstab(
    Y_Probe_test,
    Y_Probe_pred,
    rownames=["Actual attacks"],
    colnames=["Predicted attacks"],
)
Y_R2L_pred = clf_KNN_R2L.predict(X_R2L_test)
pd.crosstab(
    Y_R2L_test, Y_R2L_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_U2R_pred = clf_KNN_U2R.predict(X_U2R_test)
pd.crosstab(
    Y_U2R_test, Y_U2R_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
from sklearn.model_selection import cross_val_score
from sklearn import metrics

accuracy = cross_val_score(
    clf_KNN_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_KNN_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="precision"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_KNN_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="recall")
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_KNN_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="f1")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_KNN_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_KNN_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_KNN_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(
    clf_KNN_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="f1_macro"
)
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_KNN_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_KNN_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_KNN_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_KNN_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_KNN_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_KNN_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_KNN_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_KNN_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
from sklearn.svm import SVC

clf_SVM_DoS = SVC(kernel="linear", C=1.0, random_state=0)
clf_SVM_Probe = SVC(kernel="linear", C=1.0, random_state=0)
clf_SVM_R2L = SVC(kernel="linear", C=1.0, random_state=0)
clf_SVM_U2R = SVC(kernel="linear", C=1.0, random_state=0)
clf_SVM_DoS.fit(X_DoS, Y_DoS.astype(int))
clf_SVM_Probe.fit(X_Probe, Y_Probe.astype(int))
clf_SVM_R2L.fit(X_R2L, Y_R2L.astype(int))
clf_SVM_U2R.fit(X_U2R, Y_U2R.astype(int))
Y_DoS_pred = clf_SVM_DoS.predict(X_DoS_test)
pd.crosstab(
    Y_DoS_test, Y_DoS_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_Probe_pred = clf_SVM_Probe.predict(X_Probe_test)
pd.crosstab(
    Y_Probe_test,
    Y_Probe_pred,
    rownames=["Actual attacks"],
    colnames=["Predicted attacks"],
)
Y_R2L_pred = clf_SVM_R2L.predict(X_R2L_test)
pd.crosstab(
    Y_R2L_test, Y_R2L_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_U2R_pred = clf_SVM_U2R.predict(X_U2R_test)
pd.crosstab(
    Y_U2R_test, Y_U2R_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
from sklearn.model_selection import cross_val_score
from sklearn import metrics

accuracy = cross_val_score(
    clf_SVM_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_SVM_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="precision"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_SVM_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="recall")
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_SVM_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="f1")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_SVM_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_SVM_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_SVM_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(
    clf_SVM_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="f1_macro"
)
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_SVM_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_SVM_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_SVM_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_SVM_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_SVM_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_SVM_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_SVM_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_SVM_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
from sklearn.ensemble import VotingClassifier

clf_voting_DoS = VotingClassifier(
    estimators=[("rf", clf_DoS), ("knn", clf_KNN_DoS), ("svm", clf_SVM_DoS)],
    voting="hard",
)
clf_voting_Probe = VotingClassifier(
    estimators=[("rf", clf_Probe), ("knn", clf_KNN_Probe), ("svm", clf_SVM_Probe)],
    voting="hard",
)
clf_voting_R2L = VotingClassifier(
    estimators=[("rf", clf_R2L), ("knn", clf_KNN_R2L), ("svm", clf_SVM_R2L)],
    voting="hard",
)
clf_voting_U2R = VotingClassifier(
    estimators=[("rf", clf_U2R), ("knn", clf_KNN_U2R), ("svm", clf_SVM_U2R)],
    voting="hard",
)
clf_voting_DoS.fit(X_DoS, Y_DoS.astype(int))
clf_voting_Probe.fit(X_Probe, Y_Probe.astype(int))
clf_voting_R2L.fit(X_R2L, Y_R2L.astype(int))
clf_voting_U2R.fit(X_U2R, Y_U2R.astype(int))
Y_DoS_pred = clf_voting_DoS.predict(X_DoS_test)
pd.crosstab(
    Y_DoS_test, Y_DoS_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_Probe_pred = clf_voting_Probe.predict(X_Probe_test)
pd.crosstab(
    Y_Probe_test,
    Y_Probe_pred,
    rownames=["Actual attacks"],
    colnames=["Predicted attacks"],
)
Y_R2L_pred = clf_voting_R2L.predict(X_R2L_test)
pd.crosstab(
    Y_R2L_test, Y_R2L_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)
Y_U2R_pred = clf_voting_U2R.predict(X_U2R_test)
pd.crosstab(
    Y_U2R_test, Y_U2R_pred, rownames=["Actual attacks"], colnames=["Predicted attacks"]
)

accuracy = cross_val_score(
    clf_voting_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_voting_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="precision"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_voting_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="recall"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_voting_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring="f1")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_voting_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_voting_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_voting_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(
    clf_voting_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring="f1_macro"
)
print("F-mesaure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_voting_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_voting_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_voting_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_voting_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
accuracy = cross_val_score(
    clf_voting_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="accuracy"
)
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(
    clf_voting_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="precision_macro"
)
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(
    clf_voting_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="recall_macro"
)
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_voting_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring="f1_macro")
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
