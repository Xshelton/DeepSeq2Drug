import sys
import os
print(os.getcwd())
cdir=r'{}//contructed dataset//'.format(os.getcwd())
sys.path.append(cdir)
from pathlib import Path
import os
def check_create(folder_name):
  my_file = Path(".\{}".format(folder_name))
  if my_file.is_dir():
    print('dir exist')
  else:
    print('not exist')
    os.makedirs(my_file)
    print('making a folder, finished')
    
for frs in range(0,5):
    from folds import FiveFold_file
    check_create(r'{}//contructed dataset//5folds//'.format(os.getcwd()))
    ddir=r'{}//contructed dataset//5folds//'.format(os.getcwd())
    
    FiveFold_file(cdir,ddir,r'DrugvirusGCNMAD(62)_po781_ne781_rs{}.csv'.format(frs),6)
    #FiveFold_file(cdir,ddir,r'DrugVirus(same set)_po1156_ne1156_rs{}.csv'.format(frs),6)
