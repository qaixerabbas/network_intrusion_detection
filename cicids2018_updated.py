import numpy as np                                                                                     
import pandas as pd                                                                                    
import seaborn as sns                                                                                  
import matplotlib.pyplot as plt                                                                        
from sklearn import preprocessing                                                                      
from sklearn.preprocessing import LabelEncoder, Imputer,MinMaxScaler                                   
from sklearn.model_selection import train_test_split,cross_val_score                                   
from sklearn.ensemble import RandomForestClassifier                                                    
from sklearn.decomposition import PCA                                                                  
from sklearn.metrics import confusion_matrix,accuracy_score,precision_recall_fscore_support            
from sklearn.neighbors import KNeighborsClassifier                                                     
from sklearn.tree import DecisionTreeClassifier                                                        
from sklearn.svm import SVC                                                                            
from keras.wrappers.scikit_learn import KerasClassifier                                                
from keras.models import Sequential                                                                    
from keras.layers import Dense                                                                         
from keras.optimizers import Adam                                                                      
from keras.utils.np_utils import to_categorical                                                        
from sklearn.ensemble import RandomForestRegressor                                                     
df=pd.read_csv('combined2.csv')                                                                        
df_value=df[' Label'].value_counts()                                                                   
df[' Label']=df[' Label'].apply({'DoS Hulk':'DoS', 'DoS GoldenEye':'DoS','DoS Slowhttptest':'DoS','DoS 
slowloris':'DoS' ,'BENIGN':'BENIGN' ,'DDoS':'DDoS', 'PortScan':'PortScan'}.get)                        
df2=df.drop_duplicates()                                                                               
df2_value=df2[' Label'].value_counts()                                                                 
datatype=df2.dtypes                                                                                    
df2['Flow Bytes/s']=df2['Flow Bytes/s'].astype('float64')                                              
df2[' Flow Packets/s']=df2[' Flow Packets/s'].astype('float64')                                        
NaN_values=df2.isnull().sum()                                                                          
df2['Flow Bytes/s'].fillna(df2['Flow Bytes/s'].mean(),inplace=True)                                    
print('Datasetin ilk okunduÄŸu hali: \n',df_value)                                                     
print('Datasetin ilk (row,Column) sayÄ±sÄ±: {} '.format(df.shape))                                     
print('Datasetin Labelindeki DoS daldÄ±rÄ±larÄ±nÄ±n birleÅŸtirilmesi ve gÃ¼rÃ¼ltÃ¼nÃ¼n azaltÄ±lmasÄ±:\n',df2_value)                                                                                           
print('Datasetin son (row,Column) sayÄ±sÄ±: {} '.format(df2.shape))                                    
dataset=pd.read_csv('dataset.csv')                                                                     
dataset                                                                                                
DoS_df1=dataset[dataset[' Label']=='BENIGN']                                                           
DoS_df=DoS_df1.append(dataset[dataset[' Label']=='DoS'])                                               
DoS_df                                                                                                 
DDoS_df1=dataset[dataset[' Label']=='BENIGN']                                                          
DDoS_df=DDoS_df1.append(dataset[dataset[' Label']=='DDoS'])                                            
DDoS_df                                                                                                
PortScan_df1=dataset[dataset[' Label']=='BENIGN']                                                      
PortScan_df=PortScan_df1.append(dataset[dataset[' Label']=='PortScan'])                                
PortScan_df                                                                                            
NA_df=dataset                                                                                          
NA_df[' Label']=NA_df[' Label'].apply({'DoS':'Anormal','BENIGN':'Normal' ,'DDoS':'Anormal', 'PortScan':'Anormal'}.get)                                                                                        
NA_df                                                                                                  
def train_test_dataset(df):                                                                            
    labelencoder = LabelEncoder()                                                                      
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])                                        
    X = df.drop([' Label'],axis=1)                                                                     
    y = df.iloc[:, -1].values.reshape(-1,1)                                                            
    y=np.ravel(y)                                                                                      
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 0, stratify = y)                                                                               
    return  X_train, X_test, y_train, y_test                                                           
def RandomForest(X_train, X_test, y_train, y_test):                                                    
    rf = RandomForestClassifier(random_state = 0)                                                      
    imputer = Imputer(missing_values="NaN", strategy = "mean")                                         
    imputer = imputer.fit(X_train)                                                                     
    X_train = imputer.transform(X_train)                                                               
    X_test = imputer.transform(X_test)                                                                 
    rf.fit(X_train,y_train)                                                                            
    rf_score=rf.score(X_test,y_test)                                                                   
    y_predict=rf.predict(X_test)                                                                       
    y_true=y_test                                                                                      
    print('Random Forest Accuracy:'+ str(rf_score))                                                    
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')                                                                                                      
    print('Random Forest precision_recall_fscore:'+(str(precision))+(str(recall))+(str(fscore)))       
    cm=confusion_matrix(y_true,y_predict)                                                              
    f,ax=plt.subplots(figsize=(5,5))                                                                   
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)                           
    plt.xlabel("y_pred")                                                                               
    plt.ylabel("y_true")                                                                               
    plt.show()                                                                                         
    return rf_score,precision,recall,fscore,none                                                       
def DecisionTree(X_train, X_test, y_train, y_test):                                                    
    dt = DecisionTreeClassifier(random_state = 0)                                                      
    imputer = Imputer(missing_values="NaN", strategy = "mean")                                         
    imputer = imputer.fit(X_train)                                                                     
    X_train = imputer.transform(X_train)                                                               
    X_test = imputer.transform(X_test)                                                                 
    dt.fit(X_train, y_train)                                                                           
    score=dt.score(X_test,y_test)                                                                      
    print('Decision Tree Accuracy:'+ str(score))                                                       
    y_predict=dt.predict(X_test)                                                                       
    y_true=y_test                                                                                      
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')                                                                                                      
    print('Decision Tree precision_recall_fscore:'+(str(precision))+(str(recall))+(str(fscore)))       
    cm=confusion_matrix(y_true,y_predict)                                                              
    f,ax=plt.subplots(figsize=(5,5))                                                                   
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)                           
    plt.xlabel("y_pred")                                                                               
    plt.ylabel("y_true")                                                                               
    plt.show()                                                                                         
    return score,precision,recall,fscore,none                                                          
def kNN(X_train, X_test, y_train, y_test):                                                             
    knn=KNeighborsClassifier(n_neighbors=5)                                                            
    imputer = Imputer(missing_values="NaN", strategy = "mean")                                         
    imputer = imputer.fit(X_train)                                                                     
    X_train = imputer.transform(X_train)                                                               
    X_test = imputer.transform(X_test)                                                                 
    knn.fit(X_train,y_train)                                                                           
    prediction=knn.predict(X_test)                                                                     
    score=knn.score(X_test,y_test)                                                                     
    print("5 nn score:"+ str(score))                                                                   
    y_predict=knn.predict(X_test)                                                                      
    y_true=y_test                                                                                      
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')                                                                                                      
    print('5nn precision_recall_fscore:'+(str(precision))+(str(recall))+(str(fscore)))                 
    cm=confusion_matrix(y_true,y_predict)                                                              
    f,ax=plt.subplots(figsize=(5,5))                                                                   
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)                           
    plt.xlabel("y_pred")                                                                               
    plt.ylabel("y_true")                                                                               
    plt.show()                                                                                         
    return score,precision,recall,fscore,none                                                          
def SVM(X_train, X_test, y_train, y_test):                                                             
    svclassifier = SVC(kernel='linear')                                                                
    imputer = Imputer(missing_values="NaN", strategy = "mean")                                         
    imputer = imputer.fit(X_train)                                                                     
    X_train = imputer.transform(X_train)                                                               
    X_test = imputer.transform(X_test)                                                                 
    svclassifier.fit(X_train, y_train)                                                                 
    print("SVM Classification Accuracy:"+ str(svclassifier.score(X_test,y_test)))                      
    y_predict = svclassifier.predict(X_test)                                                           
    y_true=y_test                                                                                      
    cm=confusion_matrix(y_true,y_predict)                                                              
    f,ax=plt.subplots(figsize=(5,5))                                                                   
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)                           
    plt.xlabel("y_pred")                                                                               
    plt.ylabel("y_true")                                                                               
    plt.show()                                                                                         
def build_classifier(X_train):                                                                         
    def bm():                                                                                          
        classifier = Sequential()                                                                      
        classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))                                                                                  
        classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu'))         
                                                                                                       
        classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'softmax'))       
        lr=.003                                                                                        
        adam0=Adam(lr=lr)                                                                              
        classifier.compile(optimizer =adam0, loss = 'categorical_crossentropy', metrics = ['accuracy'])                                                                                                       
        return classifier                                                                              
    return bm                                                                                          
def ANN(X_train, X_test, y_train, y_test):                                                             
    y_ = to_categorical(y_train)                                                                       
    y_t=to_categorical(y_test)                                                                         
    estimator  = KerasClassifier(build_fn = build_classifier(X_train), epochs = 5)                     
    accuracies = cross_val_score(estimator, X = X_train, y = y_, cv = 3)                               
    mean = accuracies.mean()                                                                           
    variance = accuracies.std()                                                                        
    print("Accuracy mean: "+ str(mean))                                                                
    print("Accuracy variance: "+ str(variance))                                                        
def feature_selection(df):                                                                             
    feature=(df.drop([' Label'],axis=1)).columns.values                                                
    labelencoder = LabelEncoder()                                                                      
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])                                        
    X = df.drop([' Label'],axis=1)                                                                     
    Y = df.iloc[:, -1].values.reshape(-1,1)                                                            
    Y=np.ravel(Y)                                                                                      
    imputer = Imputer(missing_values="NaN", strategy = "mean")                                         
    imputer = imputer.fit(X)                                                                           
    X = imputer.transform(X)                                                                           
    rf = RandomForestRegressor()                                                                       
    rf.fit(X, Y)                                                                                       
    print ("Features sorted by their score:")                                                          
    print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature), reverse=True))    
feature_selection(dataset)                                                                             
DoSX_train, DoSX_test, DoSy_train, DoSy_test=train_test_dataset(DoS_df)                                
DDoSX_train, DDoSX_test, DDoSy_train, DDoSy_test=train_test_dataset(DDoS_df)                           
PS_X_train,PS_X_test,PS_y_train, PS_y_test=train_test_dataset(PortScan_df)                             
NA_X_train, NA_X_test, NA_y_train, NA_y_test=train_test_dataset(NA_df)                                 
dosrf_score,dosrf_precision,dosrf_recall,dosrf_fscore,none=RandomForest(DoSX_train, DoSX_test, DoSy_train, DoSy_test)                                                                                         
dosdt_score,dosdt_precision,dosdt_recall,dosdt_fscore,none=DecisionTree(DoSX_train, DoSX_test, DoSy_train, DoSy_test)                                                                                         
dosKnn_score,dosKnn_precision,dosKnn_recall,dosKnn_fscore,none=kNN(DoSX_train, DoSX_test, DoSy_train, DoSy_test)                                                                                              
SVM(DoSX_train, DoSX_test, DoSy_train, DoSy_test)                                                      
ANN(DoSX_train, DoSX_test, DoSy_train, DoSy_test)                                                      
labels=np.unique(DoSy_train)                                                                           
print(labels)                                                                                          
psrf_score,psrf_precision,psrf_recall,psrf_fscore,none=RandomForest(PS_X_train,PS_X_test,PS_y_train, PS_y_test)                                                                                               
psdt_score,psdt_precision,psdt_recall,psdt_fscore,none=DecisionTree(PS_X_train,PS_X_test,PS_y_train, PS_y_test)                                                                                               
psKnn_score,psKnn_precision,psKnn_recall,psKnn_fscore,none=kNN(PS_X_train,PS_X_test,PS_y_train, PS_y_test)                                                                                                    
SVM(PS_X_train,PS_X_test,PS_y_train, PS_y_test)                                                        
ANN(PS_X_train,PS_X_test,PS_y_train, PS_y_test)                                                        
ddosrf_score,ddosrf_precision,ddosrf_recall,ddosrf_fscore,none=RandomForest(DDoSX_train, DDoSX_test, DDoSy_train, DDoSy_test)                                                                                 
ddosdt_score,ddosdt_precision,ddosdt_recall,ddosdt_fscore,none=DecisionTree(DDoSX_train, DDoSX_test, DDoSy_train, DDoSy_test)                                                                                 
ddosKnn_score,ddosKnn_precision,ddosKnn_recall,ddosKnn_fscore,none=kNN(DDoSX_train, DDoSX_test, DDoSy_train, DDoSy_test)                                                                                      
SVM(DDoSX_train, DDoSX_test, DDoSy_train, DDoSy_test)                                                  
ANN(DDoSX_train, DDoSX_test, DDoSy_train, DDoSy_test)                                                  
narf_score,narf_precision,narf_recall,narf_fscore,none=RandomForest(NA_X_train, NA_X_test, NA_y_train, 
NA_y_test)                                                                                             
nadt_score,nadt_precision,nadt_recall,nadt_fscore,none=DecisionTree(NA_X_train, NA_X_test, NA_y_train, 
NA_y_test)                                                                                             
naKnn_score,naKnn_precision,naKnn_recall,naKnn_fscore,none=kNN(NA_X_train, NA_X_test, NA_y_train, NA_y_test)                                                                                                  
SVM(NA_X_train, NA_X_test, NA_y_train, NA_y_test)                                                      
ANN(NA_X_train, NA_X_test, NA_y_train, NA_y_test)                                                      
d={'Algoritmalar': ["Random Forest", "Decision Tree","KNN","ANN"],                                     
   'DoS accuracy': [dosrf_score,dosdt_score,dosKnn_score,0.7636],                                      
   'DDoS accuracy': [ddosrf_score, ddosdt_score,ddosKnn_score,0.8307],                                 
   'Port Scan accuracy':[psrf_score,psdt_score,psKnn_score,0.8738],                                    
   'Normal/Anormal accuracy':[narf_score,nadt_score,naKnn_score,0.6034],                               
  }                                                                                                    
dataframe= pd.DataFrame(data=d)                                                                        
dataframe   
