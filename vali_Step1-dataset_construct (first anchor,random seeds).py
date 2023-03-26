import pandas as pd
import random
from pathlib import Path
import os
def load_positive_samples(filename,A,B):#读取正样本
    df=pd.read_csv(filename)
    print(df.columns)
    nodeA=df['{}'.format(A)].values.tolist()
    nodeB=df['{}'.format(B)].values.tolist()
    posi_name=[]
    list_of_label=[]
    for i in range(0,len(df)):
        temp=str(nodeA[i])+'-'+str(nodeB[i])
        posi_name.append(temp)
        list_of_label.append(1)
    print((posi_name[0:10]))
    df['label']=list_of_label
    print(df)
    return df,posi_name
    

def add_positive_samples(filename,A,B,posi_name):
    print('Original',len(posi_name)) 
    df=pd.read_csv(filename)
    nodeA=df['{}'.format(A)].values.tolist()
    nodeB=df['{}'.format(B)].values.tolist()
    for i in range(0,len(df)):
        temp=str(nodeA[i])+'-'+str(nodeB[i])
        posi_name.append(temp)
        #list_of_label.append(1)
    print('New',len(posi_name))
    return posi_name


def load_ab_file(Afile,A,Bfile,B,outA,outB,posi_name,mount,random_seed):#load_file and generate enough negative samples
    dfA=pd.read_csv(Afile)
    dfB=pd.read_csv(Bfile)
    print('Afile_size',len(dfA))
    print('Bfile_size',len(dfB))
    nodeAs=dfA['{}'.format(A)].values.tolist()
    nodeBs=dfB['{}'.format(B)].values.tolist()
    #print(nodeAs,nodeBs)
    list_of_negative=[]
    random.seed(random_seed)    
    while len(list_of_negative)<=mount:
      if len(list_of_negative)%50==0:
          print(len(list_of_negative),'out of',mount)
      Ar=random.randint(0,len(dfA)-1)
      Br=random.randint(0,len(dfB)-1)
      #print(Ar,Br)
      nodeA=nodeAs[Ar]
      nodeB=nodeBs[Br]
      #print(nodeA,nodeB)
      check_if=str(nodeA)+'-'+str(nodeB)
      if check_if not in posi_name:
        list_of_negative.append({'{}'.format(outA):nodeA,'{}'.format(outB):nodeB,'label':0})#negative samples thus the 
    ln=pd.DataFrame(list_of_negative)
    print(ln)
    return ln
def check_create(folder_name):
  my_file = Path(".\{}".format(folder_name))
  if my_file.is_dir():
    print('dir exist')
  else:
    print('not exist')
    os.makedirs(my_file)
    print('making a folder, finished')
    
for random_seed in range(0,5):
    df1,posi_name=load_positive_samples('.\original dataset\drug_vrus X GCNMDA\Drugvirus_GCN.csv','Virus Name','DBID')
    posi_name=add_positive_samples('.\original dataset\CTD\CTDdataset\DrugBank(+id)_X_CTD_VirusName11_Drugs585_1592.csv','Virus Name','DBID',posi_name)
    posi_name=add_positive_samples('.\original dataset\drug_bank_with_drug_virus\DrugBank(id)_X_DrugVirus_VirusName103_Drugs190_1156.csv','Virus Name','DBID',posi_name)
    df2=load_ab_file(r'.\0_virus_feature_pool\Virus_GCNMDA62_label.csv','label',r'.\0_drug_feature_pool\Drug_GCNMDA_label.csv','label',outA='Virus Name',outB='DBID',posi_name=posi_name,mount=len(df1)-1,random_seed=random_seed)
    print(df2)
    df3=pd.concat([df1,df2],axis=0)
    print(df3.shape)
    check_create(r'.\contructed dataset')
    df3.to_csv('.\contructed dataset\DrugvirusGCNMAD(62)_po{}_ne{}_rs{}.csv'.format(len(df1),len(df2),random_seed),index=None)
