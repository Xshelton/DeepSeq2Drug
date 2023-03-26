import pandas as pd
from numpy.random import seed
from sklearn.model_selection import train_test_split

def five_fold_crossvalidation(cdir,ddir,outputname,seq,ex):#feed X into theses return five 
    #filename='hsa+neg'
    df=pd.read_csv('{}{}.csv'.format(cdir,outputname[0:-4]))
    columns=df.columns
    label=df['label']
    df=df.sample(frac=1).reset_index(drop=True)#reindex df
    X=df.values
    Y=label.values
    def details(Y):
        #print('length of Y',len(Y))
        count0=0
        count1=0
        MM=Y.reset_index(drop=True)
        MM=MM.values
        MM=MM.tolist()
        for i in range(0,len(MM)):
              #print(type(MM[i]),MM[i])
              if MM[i][0]==0:
                  #print(i,count0)
                  count0=count0+1
              if MM[i][0]==1:
                  #print(i,count0)
                  count1=count1+1
        return count0,count1
   # kf = 
    print('read_end')
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    KF=StratifiedKFold(n_splits=5, random_state=seq, shuffle=True)
    #KF=KFold(n_splits=5)  #establish KFOLD
    count=0
    for train_index,test_index in KF.split(X,Y):
      print("TRAIN:",train_index,"TEST:",test_index)
      
      X_train,X_test=X[train_index],X[test_index]
      Y_train,Y_test=Y[train_index],Y[test_index]
      X_train=pd.DataFrame(X_train)
      X_test=pd.DataFrame(X_test)
      Y_train=pd.DataFrame(Y_train)
      Y_test=pd.DataFrame(Y_test)
      #print('first_row_of_Y-test',Y_test[0])
      c1,c2=details(Y_train)
      c3,c4=details(Y_test)
      print('Y_train_details','0:',c1,'1:',c2,',Y_test_details','0:',c3,'1:',c4)#show details about the dataset
      #print(Y_test)
      count+=1
      X_train.columns=columns
      X_test.columns=columns
      if ex==1:
        X_train.to_csv('{}//{}_epoch_{}_{}.csv'.format(ddir,outputname[0:-4],seq-1,count),index=None)
        X_test.to_csv('{}//{}_epoch_{}_{}_test.csv'.format(ddir,outputname[0:-4],seq-1,count),index=None)

def FiveFold_file(cdir,ddir,filename,rep):
    for i in range(1,rep):
        outputname=r'{}'.format(filename)
        five_fold_crossvalidation(cdir,ddir,outputname,i,ex=1)
    
