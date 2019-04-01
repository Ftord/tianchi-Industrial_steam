import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFpr,f_regression
from sklearn.decomposition import PCA

def datatake():
    file = pd.read_table('/home/megnjing/下载/zhengqi_train.txt',encoding='utf-8',delim_whitespace=True)
    file_test = pd.read_table('/home/megnjing/下载/zhengqi_test.txt',encoding='utf-8',delim_whitespace=True)
    file_drop = file.drop(['V5','V9','V11','V17','V22','V28'],axis=1)
    file_test_drop = file_test.drop(['V5','V9','V11','V17','V22','V28'],axis=1)
    # method : {‘pearson’, ‘kendall’, ‘spearman’}
    corr =file_drop.corr(method="spearman")
    drop = corr[corr['target']<0.1].index

    file_af =  file_drop.drop(list(drop),axis=1)
    file_test_af = file_test_drop.drop(list(drop),axis=1)
    labels = file_af['target']
    data_features = file_af.drop(['target'],axis=1)
    data_array = data_features.values
    labels = labels.values
    fea_test = file_test_af.values
    return data_array,labels,fea_test

def kfold(X,y):
    X=np.array(X)
    y=np.array(y)
    kf=KFold(n_splits=5,shuffle=True)
    kf.get_n_splits(X)
    for train_index,test_index in kf.split(X):
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
    return X_train,X_test,y_train,y_test

def feature_select(x,y):
    vt  = SelectFpr(f_regression,alpha=0.05)
    samples_selected = vt.fit_transform(x,y)
    get_index_selected = vt.get_support(indices=True)
    return samples_selected,get_index_selected

def pca(x):
    pca = PCA(n_components='mle',svd_solver ='full')
    samples_pca = pca.fit_transform(x)
    return samples_pca

def Normalization(X):
    x_list=[]
    for line in range(len(X[:,0])):

        x_No = [(float(i)-min(X[line,:]))/float(max(X[line,:])-min(X[line,:])) for i in X[line,:]]
        x_list.extend(x_No)
    x_re = np.reshape(x_list,[len(X[:,0]),len(X[0,:])])
    return x_re

def build_model(train_data,train_label,test_data,test_label):
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
              loss='mean_squared_error',
              metrics=['accuracy'])
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    epochs = 500  #迭代次数
    model.fit(train_data, train_label, epochs=epochs,batch_size = 5,
              callbacks = callbacks,validation_data=(test_data,test_label))
    return model

def main():
    features_read,label_read,features_test_read=datatake()
    print(np.shape(features_test_read))
    samples_pca = pca(features_read)
    print(np.shape(samples_pca))
    #samples_no = Normalization(samples_pca)
    X_train,X_test,y_train,y_test = kfold(samples_pca,label_read)
    model = build_model(X_train,y_train,X_test,y_test)
    evaluate = model.evaluate(X_test,y_test,batch_size=5)
    text_pca  = pca(features_test_read)
    #text_no = Normalization(text_pca)
    result = model.predict(text_pca,batch_size=5)
    print(len(result))
    file = open("/home/megnjing/下载/zhengqi_test_labels19122_lianxi.txt",'w')
    for i in range(len(result)):
        test_labels = result[i]
        file.write(str(test_labels[0])+"\n")
    file.close()
    print(evaluate)
    print('finished')

if __name__ == '__main__':
    main()
