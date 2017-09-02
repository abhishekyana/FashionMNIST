import numpy as np
import pickle as p
from mlxtend.preprocessing import one_hot

#open and read the content of file
file='fashion-mnist_train.csv'
with open(file,'r') as f:
    data=f.read()
#Cleaning the data from csv
csdata=data.split('\n')
cdata=[]
for line in csdata[1:-1]:
    cdata.append(list(map(int,line.split(','))))
#loading into the numpy arrays
npdata=np.array(cdata[:]);del file,cdata,csdata,data;
#Seperating the labeled data
Y=npdata[:,0]
X=npdata[:,1:]
#making as per the vector notation
X_train=X.T; Y_train=one_hot(Y.reshape(Y.shape[0],)).T
 #debugging

file='fashion-mnist_test.csv'
with open(file,'r') as f:
    data=f.read()
#Cleaning the data from csv
csdata=data.split('\n')
cdata=[]
for line in csdata[1:-1]:
    cdata.append(list(map(int,line.split(','))))
#loading into the numpy arrays
npdata=np.array(cdata[:]);del file,cdata,csdata,data;
#Seperating the labeled data
Y=npdata[:,0]
X=npdata[:,1:]
#making as per the vector notation
X_test=X.T; Y_test=one_hot(Y.reshape(Y.shape[0],)).T
 #debugging
# fashion_mnist_data={"X_test":X_test,"X_train":X_train,"Y_test":Y_test,"Y_train":Y_train}
# p.dump(fashion_mnist_data,open('fmnistdata.d','wb'))

#DATA IS DUMPED WITH THE SAME NAME!!!!!!^
"""
data=p.load(open('fmnistdata.d','rb'))
data is a dictionary and can be used to obtain the values
"""
