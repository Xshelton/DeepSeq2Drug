list_r1=[1,1,0,0,0,0,0,0,0]
list_r2=[1,0,1,0,1,1,0,0,0]
list_r3=[0.1,0.2,0.3,0.5,0.5,0.5,0.05,0.05,0.05]
from sklearn.metrics import accuracy_score#从Sklearn指标中 引入准确率
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score
def load_the_model(filename):
    print(s)

def load_the_dataset(filename,label):#generate the correspoding dataset
    df=pd.read_csv(filename)
    real_label=df[label].values.tolist()
    return df,real_label
#def generate_the_real_label(): 使用已经训练好的模型，生成新的数据

def calculate_metric(list_r1,list_r2):#real/test_result
  y_test=list_r1
  y_p=list_r2
  acc = metrics.accuracy_score(y_test,y_p)
  print('ACC',acc)
  f1score=metrics.f1_score(y_test, y_p)
  print('F1_score',f1score)
  MCC=metrics.matthews_corrcoef(y_test, y_p)
  print('MCC，+1 represents a perfect prediction, 0 an average random prediction and -1 and inverse prediction',MCC)
def calculate_metric_notI(list_r1,list_r3,pos_label):
  y_test=list_r1
  y_pb=list_r3                 
  #y_pb=clf5.predict_proba(X_test)
  #print(y_pb[:,0])
  #auc=roc_auc_score(y_test, y_pb)
  #if auc<0.5:
  auc=roc_auc_score(y_test, y_pb)
  #auc=metrics.auc(y_test,y_pb)
  print('AUC',auc)
  
  macro_=metrics.average_precision_score(y_test, y_pb, average='macro', pos_label=pos_label, sample_weight=None)
  print('AUPR',macro_)
  return auc,macro_
  #if macro_<0.5:
  #    macro_=metrics.average_precision_score(y_test, y_pb[:,1], average='macro', pos_label=1, sample_weight=None)
  #precision_e,recall_e,th=metrics.precision_recall_curve(y_test, y_pb[:,0])


#generate the metric needed.
  
#calculate_metric(list_r1,list_r2)
#calculate_metric_notI(list_r1,list_r3)
