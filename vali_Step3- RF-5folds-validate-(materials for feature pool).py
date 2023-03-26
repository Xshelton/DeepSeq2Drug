 # -*- coding: utf-8 -*-
#input the embedding file name of mirna/gene
#input the positive samples and negative samples dataset
#mode1:output the embedding file for the training/validating(one file)
#mode2:output the embedding file for the training80%/validating10%/testing%(two files)
#parameters seeds for np, mirna_file,gene_file,dataset_name
import pandas as pd
from numpy.random import seed
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score#从Sklearn指标中 引入准确率
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib
global list_y_test
global list_name
global list_result
global list_color
from pathlib import Path
from trans_test import calculate_metric_notI
from trans_test import calculate_metric
global y_test_index,test_labelA,test_labelB
#import keras
list_y_test=[]
list_name=[]
list_result=[]
list_color=[]
def clear_all_list():
    global list_y_test
    global list_name
    global list_result
    global list_color
    list_y_test=[]
    list_name=[]
    list_result=[]
    list_color=[] 
clear_all_list()
def printPath(path):  
    global allFileNum 
    fileList = []
    files = os.listdir(path)
    for f in files:
        if(os.path.isfile(path + '/' + f)):  
            # 添加文件  
            fileList.append(f)
    print(fileList)
    return fileList

def roc_multiple(list_y_test,list_name,list_result,list_color):    
 print('开始载入图片')
 #plt.figure()
 #控制线的粗细
 plt.figure(figsize=(5,5))
 
 lw = 2
 print(len(list_name),len(list_result),len(list_result))
 for i in range(0,len(list_name)):
   y_test=list_y_test[i]
   fpr,tpr,threshold = roc_curve(y_test, list_result[i])
   precision,recall,thresholds = metrics.precision_recall_curve(y_test, list_result[i])
   
   roc_auc = auc(fpr,tpr)
   AUPR_=metrics.average_precision_score(y_test, list_result[i], average='macro', pos_label=1, sample_weight=None)
   plt.plot(fpr, tpr, color=list_color[i],lw=lw, label='{} (AUC = %0.2f)'.format(list_name[i]) % roc_auc)
   plt.plot(precision,recall, color=list_color[i],lw=lw, label='{} (AUPR = %0.2f)'.format(list_name[i]) % AUPR_)
 plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('Receiver operating characteristic curve')
 plt.legend(loc="lower right")
 plt.show()

           #Test_set=pd.concat([X_test,y_test],axis=1)
           #Test_set.to_csv('{}_test.csv'.format(outputname[0:-4]),index=None)#10% independent test set
def read_file(gg,filename):
   df=pd.read_csv(filename)
#gene	label	mirna
   label=df['label']
        #df=df.drop(['id'],axis=1)
   #df=df.drop(['0_mirna'],axis=1)
   if 'mirna' in df.columns.values:
    df=df.drop(['mirna'],axis=1)
    df=df.drop(['gene'],axis=1)
    df=df.drop(['label'],axis=1)
   if '0_mirna' in df.columns.values:
    df=df.drop(['0_mirna'],axis=1)
    df=df.drop(['1_gene'],axis=1)
    df=df.drop(['label'],axis=1)
   list_label2=[]
   for i in range(0,len(label)):
      if label[i]==1:
             list_label2.append(0)
      else:
             list_label2.append(1)
   list_2_pd=pd.DataFrame(list_label2)
   #label=pd.concat([label,list_2_pd],axis=1)
   Y = keras.utils.to_categorical(label)
   X_train, X_test, y_train, y_test,Y_train_onehot,y_test_onehot = train_test_split(df,label,Y,test_size=0.2, random_state=0)
   print('length of trainx',len(X_train))
   print('length of trainY',len(y_train))
   print('length of testX',len(X_test))
   print('length of testY',len(y_test))
   return X_train, X_test, y_train, y_test,Y_train_onehot,y_test_onehot,df,label

def append_list(name,y_test,y_predict,color):
    global list_y_test
    global list_name
    global list_result
    global list_color
    list_y_test.append(y_test)
    list_name.append(name)
    list_result.append(y_predict)
    list_color.append(color)
    
def Scaler(X_train,X_test):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
   # X_train=X_train.drop(['mirna'],axis=1)
   # X_test=X_test.drop(['mirna'],axis=1)
   # X_train=X_train.drop(['gene'],axis=1)
   # X_test=X_test.drop(['gene'],axis=1)
   # X_train=X_train.drop(['label'],axis=1)
    #X_test=X_test.drop(['label'],axis=1)
    X_train = scaler.fit_transform(X_train)
    #print(train_xgmmp.shape)
    X_test=scaler.transform(X_test)
    return X_train,X_test

def RF(X_train,y_train,X_test,y_test):#0.52
  from sklearn.ensemble import RandomForestClassifier
  clf5 = RandomForestClassifier(n_estimators=120)#
  #scaler = MinMaxScaler()
  scaler=StandardScaler()
  scaler.fit(X_train)
  X_train=scaler.transform(X_train)
  X_test=scaler.transform(X_test)
  clf5.fit(X_train,y_train)
  y_p=clf5.predict(X_test)
  acc = metrics.accuracy_score(y_test,y_p)
  print('RF',acc)
  y_pb=clf5.predict_proba(X_test)
  #print(y_pb[:,0])
  auc=roc_auc_score(y_test, y_pb[:,0])
  if auc<0.5:
      auc=roc_auc_score(y_test, y_pb[:,1])
 # auc=metrics.auc(y_test,y_pb[:,0])
  print('RF_AUC',auc)
  f1score=metrics.f1_score(y_test, y_p)
  print('RF_F1',f1score)
  MCC=metrics.matthews_corrcoef(y_test, y_p)
  print('RF MCC',MCC)
  macro_=metrics.average_precision_score(y_test, y_pb[:,0], average='macro', pos_label=1, sample_weight=None)
  if macro_<0.5:
      macro_=metrics.average_precision_score(y_test, y_pb[:,1], average='macro', pos_label=1, sample_weight=None)
  precision_e,recall_e,th=metrics.precision_recall_curve(y_test, y_pb[:,0])
  plt.plot(recall_e,precision_e, color = 'red')
  print("RF AUPR", macro_)
  print('RF:',metrics.classification_report(y_test,y_p))
  name='RF'
  #y_predict=y_score
  color='orange'
  append_list(name,y_test,y_p,color)
  return auc,macro_,f1score,MCC,metrics.classification_report(y_test,y_p)
def RF_5fold(X_train,Y_train,X_test,Y_test,rs,valid_model,valid_result,epoch):
        from sklearn import model_selection
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        global y_test_index,test_labelA,test_labelB
        #vali_model_dir
        # if model exist
        Retrain_flag=False
        #if Retrain_flage==True or mo
        if os.path.exists(valid_model) and Retrain_flag==False: 
            #load model
            print('model exist, try to load model and predict')
            scaler = StandardScaler()
            X_train=scaler.fit_transform(X_train)
            #rs=0
            X_test=scaler.transform(X_test)
            gsearch1 = joblib.load('{}'.format(valid_model)) #调用
            Y_predict=gsearch1.predict(X_test)
            y_predict_prob=gsearch1.predict_proba(X_test)
            auc,aupr=calculate_metric_notI(Y_test,y_predict_prob[:,1],1)
            f1score=metrics.f1_score(Y_test,Y_predict)
      #print('fine_tunde_LR_F1',f1score)
            MCC=metrics.matthews_corrcoef(Y_test, Y_predict)
            key={'test_auc':auc,'test_aupr':aupr,'f1':f1score,'MCC':MCC,'random_state':rs,'best_para':gsearch1.best_params_,'model':'RF-5fold','epoch':epoch}
            DF_Ypredict=pd.DataFrame(Y_predict)
            DF_Ypredict=DF_Ypredict.rename(columns={'0':'Predict_label'})
            DF_Ypredict['real_label']=Y_test.values
            DF_Ypredict['predict_prob']=y_predict_prob[:,1]
           
            #DF_Ypredict['original index']=y_test_index
            DF_Ypredict['labelA']=test_labelA.values
            DF_Ypredict['labelB']=test_labelB.values
            DF_Ypredict.to_csv('{}_regenerate.csv'.format(valid_result[0:-4]),index=None)
            
            return auc,aupr,key
        else:
            print('model not exist','generate mode')
            scaler = StandardScaler()
            X_train=scaler.fit_transform(X_train)
            #rs=0
            X_test=scaler.transform(X_test)
            #set paraset
            param_test1 = {'n_estimators':range(20,150,10),'min_samples_leaf':range(5,50,10),}#'max_depth':range(4,15,1)}#120,20,8
            gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,max_depth=8,max_features='sqrt' ,random_state=rs), 
                                   param_grid = param_test1, scoring='roc_auc',cv=5)
            gsearch1.fit(X_train,Y_train.values.ravel())#add .ravel() could remove warning.
            #print('The results of the test_set')
            Y_predict=gsearch1.predict(X_test)
            y_predict_prob=gsearch1.predict_proba(X_test)
            calculate_metric(Y_test,Y_predict)
            
            
            #results save
            print('model_training success')
            joblib.dump(gsearch1,valid_model)
            auc,aupr=calculate_metric_notI(Y_test,y_predict_prob[:,1],1)
            f1score=metrics.f1_score(Y_test,Y_predict)
      #print('fine_tunde_LR_F1',f1score)
            MCC=metrics.matthews_corrcoef(Y_test, Y_predict)
      #print('fine_tunde_LR MCC',MCC)
            
            DF_Ypredict=pd.DataFrame(Y_predict)
            DF_Ypredict=DF_Ypredict.rename(columns={'0':'Predict_label'})
            DF_Ypredict['real_label']=Y_test.values
            DF_Ypredict['predict_prob']=y_predict_prob[:,1]
            #DF_Ypredict['original index']=y_test_index
            DF_Ypredict['labelA']=test_labelA.values
            DF_Ypredict['labelB']=test_labelB.values
            DF_Ypredict.to_csv(valid_result,index=None)
            key={'test_auc':auc,'test_aupr':aupr,'f1':f1score,'MCC':MCC,'random_state':rs,'best_para':gsearch1.best_params_,'model':'RF-5fold','epoch':epoch}
            #list_test_auc.append(auc)
            #list_test_aupr.append(aupr)
            return auc,aupr,key
def LR(X_train,y_train,X_test,y_test):#0.52
  import sklearn
  from numpy import logspace
  try:
    clf5 = sklearn.linear_model.LogisticRegressionCV(Cs=logspace(-2,2,3), cv=5, class_weight='unbalanced', solver='liblinear')
  except:
      from sklearn.linear_model import LogisticRegressionCV
      clf5=LogisticRegressionCV(Cs=logspace(-2,2,3), cv=5, class_weight='unbalanced', solver='liblinear')
  clf5.fit(X_train,y_train)
  y_p=clf5.predict(X_test)
  acc = metrics.accuracy_score(y_test,y_p)
  print('RF',acc)
  y_pb=clf5.predict_proba(X_test)
  #print(y_pb[:,0])
  auc=roc_auc_score(y_test, y_pb[:,0])
  if auc<0.5:
      auc=roc_auc_score(y_test, y_pb[:,1])
 # auc=metrics.auc(y_test,y_pb[:,0])
  print('fine_tuned_LR_AUC',auc)
  f1score=metrics.f1_score(y_test, y_p)
  print('fine_tunde_LR_F1',f1score)
  MCC=metrics.matthews_corrcoef(y_test, y_p)
  print('fine_tunde_LR MCC',MCC)
  macro_=metrics.average_precision_score(y_test, y_pb[:,0], average='macro', pos_label=1, sample_weight=None)
  if macro_<0.5:
      macro_=metrics.average_precision_score(y_test, y_pb[:,1], average='macro', pos_label=1, sample_weight=None)
  precision_e,recall_e,th=metrics.precision_recall_curve(y_test, y_pb[:,0])
  
  plt.plot(recall_e,precision_e, color = 'red')
  print("fine_tunde_LR AUPR", macro_)
  print('fine_tunde_LR:',metrics.classification_report(y_test,y_p))
  name='fine_tunde_LR'
  #y_predict=y_score
  color='orange'
  append_list(name,y_test,y_p,color)
  return auc,macro_,f1score,MCC,metrics.classification_report(y_test,y_p)

def ada_5fold(X_train,y_train,X_test,y_test):
    from numpy import logspace
    from numpy import array
    from sklearn import model_selection
    from sklearn import ensemble
    paramgrid={'learning_rate':logspace(-5,5,10),'n_estimators':array([7,10,13,15,17,19])}
    #print(paramgrid)# setup the cross-validation object
    adacv=model_selection.GridSearchCV(ensemble.AdaBoostClassifier(random_state=0),paramgrid,cv=3,n_jobs=5,verbose=True)# run cross-validation (train for each split)
    adacv.fit(X_train,y_train);
    print("best params for adaboost:",adacv.best_params_)
    predY_ada= adacv.predict(X_test)
    print('shape',predY_ada.shape)
    auc=roc_auc_score(y_test, predY_ada)
 # auc=metrics.auc(y_test,y_pb[:,0])
    print('fine_tuned_LR_AUC',auc)
    f1score=metrics.f1_score(y_test, predY_ada)
    print('fine_tunde_LR_F1',f1score)
    MCC=metrics.matthews_corrcoef(y_test, predY_ada)
    print('fine_tunde_LR MCC',MCC)
    macro_=metrics.average_precision_score(y_test, predY_ada, average='macro', pos_label=1, sample_weight=None)
    print("fine_tunde_LR AUPR", macro_)
    acc_ada=metrics.accuracy_score(y_test,predY_ada)
    print("test accuracy for rf =",acc_ada)#0.6641
    print('Adaboost-5fold',metrics.classification_report(y_test,predY_ada))
    name='Adaboost-5fold'
    color='green'
    append_list(name,y_test,predY_ada,color)
    return auc,macro_,f1score,MCC,metrics.classification_report(y_test,predY_ada),adacv.best_params_
def count_0_1(y_train):
    y_tl=y_train.values.tolist()
    dict1 = {}
    for key in y_tl:
       dict1[key] = dict1.get(key, 0) + 1
    print(dict1)
def embedding2dataset(fileA,labelA,fileB,labelB,dataset,dataA,dataB,outputname,partion,randomseed):
    global y_test_index,test_labelA,test_labelB
    seed(randomseed)
    dfA=pd.read_csv(fileA)
    dfA_label=dfA[labelA].values.tolist()#fileA的标签 labels of file A
    dfA=dfA.drop([labelA],axis=1)#fileA的特征 features of fileA
    dfB=pd.read_csv(fileB)
    dfB_label=dfB[labelB].values.tolist()
    dfB=dfB.drop([labelB],axis=1)
    print('step1,successfully load embedding data file for mirna/gene')
    #try:
    df_dataset=pd.read_csv(dataset,encoding='gbk')
    D_A=df_dataset[dataA].values.tolist()
    D_B=df_dataset[dataB].values.tolist()
    D_label=df_dataset['label']
    print('step2,successfully load dataset for linc/gene/label')
    #except:
    #print('dataset load error:make sure the dataset with /{}/{}/label as columns header'.format(dataA,dataB))
    miss=0
    flag=0
    list_label=[]
    print('concating and generating the dataset,begin...')
    for i in range(0,len(df_dataset)):
        A_name=D_A[i]
        B_name=D_B[i]
        if A_name in dfA_label and B_name in dfB_label:
              A_index=dfA_label.index(A_name)
              B_index=dfB_label.index(B_name)
              featureA=dfA[A_index:A_index+1]
              featureA=featureA.reset_index(drop=True)
              featureB=dfB[B_index:B_index+1]
              featureB=featureB.reset_index(drop=True)
              if D_label[i]>0:
                         list_label.append({'{}'.format(dataA):A_name,'{}'.format(dataB):B_name,'label':1})
              else:
                         list_label.append({'{}'.format(dataA):A_name,'{}'.format(dataB):B_name,'label':0})
              if flag==0:#means the first time
                       temp0=pd.concat([featureA,featureB],axis=1,join='outer')
                       flag=1
              else:    #means the second time
                       temp1=pd.concat([featureA,featureB],axis=1,join='outer')
                       temp0=pd.concat([temp0,temp1],axis=0)
              if i%500==0 and i!=0:
                      if len(temp0)!=None:
                           print('Total generation',len(temp0),'/',len(D_A),'miss number',miss)
        else:
              miss+=1
              if miss!=0 and miss%500==0:
                      print('Miss_pairs_number_milestone',miss, '{}'.format(dataA),D_A[i],'{}'.format(dataB),D_B[i])
    print('concating and generating the dataset,finshed....','Total generation',len(temp0),'/',len(D_A))
    print('begin to cut dataset')
    fea_label=pd.DataFrame(list_label)
    X=temp0
    Y=fea_label
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=partion, stratify=Y,random_state=randomseed)
    Y_index=range(len(Y))
    X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(X, Y_index,test_size=partion, stratify=Y,random_state=randomseed)
    X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(X, Y_index,test_size=partion,stratify=Y,random_state=randomseed)
    X_train_index, X_test_index, tlabelA, test_labelA = train_test_split(X, dfA_label,test_size=partion,stratify=Y,random_state=randomseed)
    X_train_index, X_test_index, tlabelB, test_labelB = train_test_split(X, dfB_label,test_size=partion,stratify=Y,random_state=randomseed)
    #ONLY INDEX IS NOT ENOUGH, BETTER SHOW THE VIRUS AND DRUG
    
    #print('see if the index is correctly generated',(X_train==X_train_index).all())
    
    y_train=y_train.drop(['{}'.format(dataA)],axis=1)
    y_test=y_test.drop(['{}'.format(dataA)],axis=1)
    y_train=y_train.drop(['{}'.format(dataB)],axis=1)
    y_test=y_test.drop(['{}'.format(dataB)],axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    print('generate_file_for_future use')
    temp0=temp0.reset_index(drop=True)
    temp0=pd.concat([temp0,fea_label],axis=1)#这个feature 就包含了所有的抽取的正样本
    temp0.to_csv(outputname,index=None)
    return X_train, y_train, X_test, y_test
def embedding2dataset_fromFILE(fileA,labelA,fileB,labelB,dataset,dataA,dataB,outputname,partion,randomseed,filetype):
    global test_labelA,test_labelB
    seed(randomseed)
    dfA=pd.read_csv(fileA)
    dfA_label=dfA[labelA].values.tolist()#fileA的标签 labels of file A
    dfA=dfA.drop([labelA],axis=1)#fileA的特征 features of fileA
    dfB=pd.read_csv(fileB)
    dfB_label=dfB[labelB].values.tolist()
    dfB=dfB.drop([labelB],axis=1)
    print('step1,successfully load embedding data file for virus/Drug')
    #try:
    df_dataset=pd.read_csv(dataset,encoding='gbk')
    D_A=df_dataset[dataA].values.tolist()
    D_B=df_dataset[dataB].values.tolist()
    D_label=df_dataset['label']
    print('step2,successfully load dataset for Virus/Drug/label')
    #except:
    #print('dataset load error:make sure the dataset with /{}/{}/label as columns header'.format(dataA,dataB))
    miss=0
    flag=0
    list_label=[]
    print('concating and generating the dataset,begin...')
    for i in range(0,len(df_dataset)):
        A_name=D_A[i]
        B_name=D_B[i]
        if A_name in dfA_label and B_name in dfB_label:
              A_index=dfA_label.index(A_name)
              B_index=dfB_label.index(B_name)
              featureA=dfA[A_index:A_index+1]
              featureA=featureA.reset_index(drop=True)
              featureB=dfB[B_index:B_index+1]
              featureB=featureB.reset_index(drop=True)
              if D_label[i]>0:
                         list_label.append({'{}'.format(dataA):A_name,'{}'.format(dataB):B_name,'label':1})
              else:
                         list_label.append({'{}'.format(dataA):A_name,'{}'.format(dataB):B_name,'label':0})
              if flag==0:#means the first time
                       temp0=pd.concat([featureA,featureB],axis=1,join='outer')
                       flag=1
              else:    #means the second time
                       temp1=pd.concat([featureA,featureB],axis=1,join='outer')
                       temp0=pd.concat([temp0,temp1],axis=0)
              if i%500==0 and i!=0:
                      if len(temp0)!=None:
                           print('Total generation',len(temp0),'/',len(D_A),'miss number',miss)
        else:
              miss+=1
              if miss!=0 and miss%500==0:
                      print('Miss_pairs_number_milestone',miss, '{}'.format(dataA),D_A[i],'{}'.format(dataB),D_B[i])
    print('concating and generating the dataset,finshed....','Total generation',len(temp0),'/',len(D_A))
    print('begin to cut dataset')
    fea_label=pd.DataFrame(list_label)
    X_train=temp0
    y_train=fea_label
    if filetype=='Test':
         #y_test_index,
         test_labelA=y_train['{}'.format(dataA)]
         test_labelB=y_train['{}'.format(dataB)]
    y_train=y_train.drop(['{}'.format(dataA)],axis=1)
    y_train=y_train.drop(['{}'.format(dataB)],axis=1)
    print('seeeee y_train',y_train)
    print('generate_file_for_future use')
    temp0=temp0.reset_index(drop=True)
    temp0=pd.concat([temp0,fea_label],axis=1)#这个feature 就包含了所有的抽取的正样本
    temp0.to_csv(outputname,index=None)
    return X_train, y_train

def file_to_dataset(dataset_name,label1,label2,partion,rs):
   from numpy import random
   global y_test_index,test_labelA,test_labelB
   #random(0)
   randomseed=0
   seed(randomseed)
   df=pd.read_csv('{}'.format(dataset_name))
   print('successfully load \'embedding\' data file for Virus/Drug')
   Y=df['label']
   labelA=df['{}'.format(label1)]
   labelB=df['{}'.format(label2)]
   df=df.drop(['{}'.format(label1),'{}'.format(label2),'label'],axis=1)

   X=df
   X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=partion,stratify=Y,random_state=rs,)#fi
   Y_index=range(len(Y))
   X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(X, Y_index,test_size=partion,stratify=Y,random_state=randomseed)
   X_train_index, X_test_index, tlabelA, test_labelA = train_test_split(X, labelA,test_size=partion,stratify=Y,random_state=randomseed)
   X_train_index, X_test_index, tlabelB, test_labelB = train_test_split(X, labelB,test_size=partion,stratify=Y,random_state=randomseed)
   print('the length of train_x',len(X_train))
   print('train_y',count_0_1(y_train))
   print('the length of test_x',len(X_test))
   print('test_y',count_0_1(y_test))
   #Train_set=pd.concat([X_train,y_train],axis=1)#contains 90% of the training set
   #Train_set.to_csv(outputname,index=None)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test=scaler.transform(X_test)
   print(type(X_train),'type of x_train')
   xtrainpd=pd.DataFrame(X_train)
   #xtrainpd.to_csv('temp_train.csv')
   return X_train,y_train,X_test,y_test


def file_to_dataset_5fold_from(dataset_name,label1,label2,filetype):
     global y_test_index,test_labelA,test_labelB
     df=pd.read_csv('{}'.format(dataset_name))
     print('successfully load \'embedding\' FIVE FOLD constructed embedding {} file for Virus/Drug'.format(filetype))
     Y=df['label']
     labelA=df['{}'.format(label1)]
     labelB=df['{}'.format(label2)]
     df=df.drop(['{}'.format(label1),'{}'.format(label2),'label'],axis=1)
     #don't need to scaler
     if filetype=='Test':
         test_labelA=labelA
         test_labelB=labelB
     return df,Y
    
def return_vdkey():
    list_vkey=['GCNMDA62']
    list_dkey=['GCNMDA']#'role2vec'
    list_vdkey=[]
    for vkey in list_vkey:
        for dkey in list_dkey:
            vdkeypair='{}-{}'.format(vkey,dkey)
            list_vdkey.append(vdkeypair)
    print(list_vdkey)
    return list_vdkey
    #['doc2vec(average_from_BD)-resnet50','-role2vec','doc2vec(average_from_BD)-bank_GPT2(768)','doc2vec-resnet50','doc2vec-role2vec','doc2vec-bank_GPT2(768)']

for frs in range(0,5):
    rep=5#control repeat times
    por=1#conrol how many part for train
    #test_1
    test_TYPE='Type2'
    fivefold_source='file'#solid file
    list_auc=[]
    list_pr=[]
    list_rf=[]
    #indicate the directory of virus features
    virus_directory=r'.\0_virus_feature_pool'
    virus_fileList=printPath(r'.\0_virus_feature_pool')
    #indicate the directory of drugs features
    drug_directory=r'.\0_drug_feature_pool'
    drug_fileList=printPath(r'.\0_drug_feature_pool')
    
    #dataset='{}drugvirusGCNMAD'.format(frs)#real negative real positive @ this is an unbalanced dataset
    dataset='{}drugvirusGCNMAD'.format(frs)
    if dataset=='{}drugvirusGCNMAD'.format(frs):
         dataset_location='.\contructed dataset\DrugvirusGCNMAD(62)_po781_ne781_rs{}.csv'.format(frs)        
    if dataset=='{}drugvirus'.format(frs):
        dataset_location=r'.\contructed dataset\DrugVirus_po1156_ne1156_rs{}.csv'.format(frs)
    if dataset=='{}CTDdrugs'.format(frs):
         dataset_location=r'.\contructed dataset\CTDdrugs_po1592_ne1592_rs{}.csv'.format(frs)
    if dataset=='{}drugvirus(SameSet)'.format(frs):
         dataset_location=r'.\contructed dataset\DrugVirus(same set)_po1156_ne1156_rs{}.csv'.format(frs)

    def check_create(path_name):
      my_file = Path(path_name)
      if my_file.is_dir():
        print('dir exist')
      else:
        print('not exist')
        os.makedirs(my_file)
        print('making a folder, finished')

    for i in range(0,len(virus_fileList)):
         #make folder for dataset/
         check_create('.\\Results\\{}\\{}_feature_pod'.format(test_TYPE,dataset))
         check_create('.\\Results\\{}\\{}_feature_pod\\5folds'.format(test_TYPE,dataset))
         check_create('.\\Embedding dataset\\{}_feature_pod'.format(dataset))
         check_create('.\\Embedding dataset\\{}_feature_pod\\5folds'.format(dataset))
         #make folder for validation model and validation results/ for AUC/AUPR values generation.
         valid_model_dir=r'.\\valid_model\\{}_{}'.format(dataset,test_TYPE)
         #dataset,
         valid_results_dir=r'.\\valid_results\\{}_{}'.format(dataset,test_TYPE)
         check_create( valid_model_dir)
         check_create(valid_results_dir)
         print(dataset_location,os.path.exists(dataset_location))
         if os.path.exists(dataset_location):#if dataset exists
             virus_doc=virus_fileList[i]
             vpre= virus_doc.find('Virus_')
             vafter=virus_doc.find('_label.csv')
             vkey=virus_doc[vpre+6:vafter]
             print('virus_faeture_key_words',vkey)
             virus_real_path = os.path.join(virus_directory,virus_doc)
             fileA=virus_real_path
             #print(fileA)
 
             for j in range(0,len(drug_fileList)):
                 list_auc=[]
                 list_pr=[]
                 list_rf=[]
                 drug_doc=drug_fileList[j]
                 #print(doc)
                 dpre=drug_doc.find('Drug_')
                 dafter=drug_doc.find('_label.csv')
                 dkey=drug_doc[dpre+5:dafter]
                 print('drug_faeture_keywords',dkey)
                 drug_real_path = os.path.join(drug_directory,drug_doc)
                 fileB=drug_real_path
                 #print(fileA,fileB)
                 #vdkey=[]
                 vdkey=return_vdkey()
                 vdkeypair='{}-{}'.format(vkey,dkey)
                 if vdkeypair in vdkey:
                     print('vdkey in,calculate')
                     for rs in range(0,rep):
                        step_para=1
                        if  fivefold_source=='file':
                            step_para=5
                        for k in range(por,por+step_para):#for 2.7 only ,k from 1-6, For other version k here control the size of the training/testing/
                              #frs file number (for negative sampling)
                              # i read virus feature pool
                              # j read drugs feature pool
                              # rs control random seeds
                              # k control training size/ for fivefold from file control test set/
                              #'DrugvirusGCNMAD_po781_ne781_rs{frs}_epoch_{rs}_{k}.csv'
                              
                              if fivefold_source=='file':
                                  print('calculate five fold cross validation from files')
                                  #dataset_location='.\contructed dataset\DrugvirusGCNMAD_po781_ne781_rs{}.csv'
                                  qindex=dataset_location.rfind('\\')
                                  fn=dataset_location[qindex+1:-4]
                                  train_file='{}_epoch_{}_{}.csv'.format(fn,rs,k)
                                  test_file='{}_epoch_{}_{}_test.csv'.format(fn,rs,k)
                                  train_dataset_location=r'.\contructed dataset\5folds\{}'.format(train_file)
                                  test_dataset_location=r'.\contructed dataset\5folds\{}'.format(test_file)
                                  train_embedding_dataset=r'.\\Embedding dataset\\{}_feature_pod\\5folds\\{}_v{}_d{}_frs_{}_rs{}_epoch{}.csv'.format(dataset,dataset,vkey,dkey,frs,rs,k)
                                  test_embedding_dataset=r'.\\Embedding dataset\\{}_feature_pod\\5folds\\{}_v{}_d{}_frs_{}_rs{}_epoch{}_test.csv'.format(dataset,dataset,vkey,dkey,frs,rs,k)
                                  try:
                                      X_train,y_train=file_to_dataset_5fold_from(dataset_name=train_embedding_dataset,label1='Virus Name',label2='DBID',filetype='Train')
                                      X_test,y_test=file_to_dataset_5fold_from(dataset_name=test_embedding_dataset,label1='Virus Name',label2='DBID',filetype='Test')
                                      
                                  except:
                                      print('load file from embedding dataset fail, begain to generate files')
                                      X_train,y_train=embedding2dataset_fromFILE(fileA=fileA,#lnc-palindormic_kernelpolyPCA64).csv
                                                                                  labelA='label',
                                                                                   fileB=fileB,
                                                                                   labelB='label',
                                                                                   dataset=train_dataset_location,
                                                                                   dataA='Virus Name',
                                                                                   dataB='DBID',
                                                                                   outputname= train_embedding_dataset,
                                                                                   partion=0,
                                                                                   randomseed=0,
                                                                                   filetype='Train',
                                                                                   )
                                      X_test,y_test=embedding2dataset_fromFILE(fileA=fileA,#lnc-palindormic_kernelpolyPCA64).csv
                                                                                  labelA='label',
                                                                                   fileB=fileB,
                                                                                   labelB='label',
                                                                                   dataset=test_dataset_location,
                                                                                   dataA='Virus Name',
                                                                                   dataB='DBID',
                                                                                   outputname=test_embedding_dataset,
                                                                                   partion=0,
                                                                                   randomseed=0,
                                                                                   filetype='Test',
                                                                                   )
                                  
                                  
                              else:
                                try:
                                 print('try to load dataset for training')#rnrp_nega465_posi2459
                                 X_train,y_train,X_test,y_test=file_to_dataset('.\\Embedding dataset\\{}_feature_pod\\\\{}_v{}_d{}_rs_{}.csv'.format(dataset,dataset,vkey,dkey,frs),'Virus Name','DBID',0.1+0.1*k,rs)
                                 
                                except:
                                 print('load_file_fail, try to generate file')#plalindrome_all_gene60_(kernel,PCA64).csv
                                     #fileA,labelA,fileB,labelB,dataset,dataA,dataB,outputname,partion,randomseed
                                 X_train,y_train,X_test,y_test=embedding2dataset(fileA=fileA,#lnc-palindormic_kernelpolyPCA64).csv
                                                                                  labelA='label',
                                                                                   fileB=fileB,
                                                                                   labelB='label',
                                                                                   dataset=dataset_location,
                                                                                   dataA='Virus Name',
                                                                                   dataB='DBID',
                                                                                   outputname='.\\Embedding dataset\\{}_feature_pod\\{}_v{}_d{}_rs_{}.csv'.format(dataset,dataset,vkey,dkey,frs),
                                                                                   partion=0.1+0.1*k,
                                                                                   randomseed=rs,
                                                                                   )
                              if test_TYPE=='Type1':
                                  auc,macro_,f1score,MCC,report=RF(X_train,y_train,X_test,y_test)
                                  lname='RF120'
                                  #auc,macro_,f1score,MCC,report=LR(X_train,y_train,X_test,y_test)
                                  #lname='LR'
                                  #auc,macro_,f1score,MCC,report,best=ada_5fold(X_train,y_train,X_test,y_test)
                                  #lname='Ada'
                                  list_auc.append(auc)
                                  list_pr.append(macro_)
                                  rf_key={'virus_feature':vkey,'drug_faeture':dkey,'AUC':auc,'AUPR':macro_,'f1':f1score,'MCC':MCC,'portion_for_train':(1-0.1-0.1*k),'model':lname}#'report':report,
                              if test_TYPE=='Type2':
                                    # add 2 parameters vali_model, valid_results,
                                    #save results name should contain frs,i,j,rs
                                    #
                                   valid_model=r'.\\{}\\{}_frs{}_V{}_D{}_RS{}_epoch{}.m'.format(valid_model_dir,dataset,frs,vkey,dkey,rs,k)
                                   valid_result=r'.\\{}\\{}_frs{}_V{}_D{}_RS{}_epoch{}.csv'.format( valid_results_dir,dataset,frs,vkey,dkey,rs,k)
                                   auc,macro_,rf_key=RF_5fold(X_train,y_train,X_test,y_test,rs,valid_model,valid_result,epoch=k)
                                   lname='RF5-fold-from-file'
                                   list_auc.append(auc)
                                   list_pr.append(macro_)
                                   list_rf.append(rf_key)
                     
                     lrf=pd.DataFrame(list_rf)
                     print('temp save')
                     from numpy import *
                     m1=round(mean(list_auc),5)
                     m2=round(mean(list_pr),5)
                     print(mean(list_auc),mean(list_pr))
                     if fivefold_source=='file':
                         lrf.to_csv('.\\Results\\{}\\{}_feature_pod\\5folds\\{}_{}_m{}_v{}_d{}_frs{}_rep{}_(5fold)train_mAUC{}_mAUPR{}.csv'.format(test_TYPE,dataset,test_TYPE,dataset,lname,vkey,dkey,frs,rep,m1,m2),index=None)
                     else:
                         lrf.to_csv('.\\Results\\{}\\{}_feature_pod\\{}_{}_m{}_v{}_d{}_frs{}_rep{}_{}train_mAUC{}_mAUPR{}.csv'.format(test_TYPE,dataset,test_TYPE,dataset,lname,vkey,dkey,frs,rep,1-0.1-0.1*por,m1,m2),index=None)
            
                 else:
                     print('vdkey not in list')

