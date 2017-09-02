import pickle as p #to dump the parameters for later use
import numpy as np #Computing library
import matplotlib.pyplot as plt #to plot the error(Cost).

data=p.load(open('fmnistdata.d','rb')) #Fashion MNIST dataset downloaded from Kaggle and dumped as ".d" and loaded it here.
X,Y=data["X_train"],data["Y_train"] #One-hot encoded Y.
np.random.seed(6) #For consistent results

#Activation functions
def g(type,z,deriv=False):
    if type=='sigmoid':
        if deriv==True:
            return z*(1-z)
        return 1/(1+np.exp(-1*z))
    if type=='relu':
        if deriv==True:
            return z>=0
        return np.maximum(0,z)
#for random normal initialization 
def initialize_params(n_x,n_h,n_y):
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.random.randn(n_h,1)*0.01
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.random.randn(n_y,1)*0.01
    params={"W1":W1,"W2":W2,"b1":b1,"b2":b2}
    return params

def forward_propagation(X,params):
    W1=params["W1"]
    b1=params["b1"]
    W2=params["W2"]
    b2=params["b2"]
    Z1=np.dot(W1,X)+b1
    A1=g('relu',Z1)
    Z2=np.dot(W2,A1)+b2
    A2=1/(1+np.exp(-Z2))
    cache={'Z1':Z1,'Z2':Z2,'A1':A1,'A2':A2}
    return A2,cache

def back_propagation(X,Y,cache,params):
    A2=cache["A2"]
    A1=cache["A1"]
    Z2=cache["Z2"]
    Z1=cache["Z1"]
    W1=params["W1"]
    W2=params["W2"]
    b1=params["b1"]
    b2=params["b2"]
    m=Y.shape[1]
    dZ2=A2-Y
    dW2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1=np.dot(W2.T,dZ2)*g('relu',Z1,True)
    dW1=np.dot(dZ1,X.T)/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m
    grads={"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2}
    return grads

#Threshold is set to 0.5(50% probability) P(Y=1|X).
def predict(X,params):
    A2,cache=forward_propagation(X,params)
    return A2>0.5

#Derivatives obtaoned with backpropagation
def update_params(params,grads,learning_rate=0.01):
    dW1=grads["dW1"]
    dW2=grads["dW2"]
    db1=grads["db1"]
    db2=grads["db2"]
    W1=params["W1"]
    W2=params["W2"]
    b1=params["b1"]
    b2=params["b2"]
    W1=W1-learning_rate*dW1
    W2=W2-learning_rate*dW2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2
    params={"W1":W1,"W2":W2,"b1":b1,"b2":b2}
    return params


def compute_cost(A2,Y):
    m=Y.shape[1]
    logprobs = np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))
    cost = -np.sum(logprobs)/m
    cost = np.squeeze(cost)
    return cost

#Putting All togather.
def model(X,Y,iters=1000,learning_rate=0.01):
    params=initialize_params(X.shape[0],1000,Y.shape[0])
    eror=[]
    for i in range(iters):
        A2,cache=forward_propagation(X,params)
        grads=back_propagation(X,Y,cache,params)
        cost=compute_cost(A2,Y)
        params=update_params(params,grads,learning_rate)
        print(f"iteration : {i} Cost :{cost}")
        eror.append(cost)
    return params,eror

#For debugging trained for first #100 batch data.
x,y=X[:,0:100]/255,Y[:,0:100]

def mainfun(x,y):
    iterss=1000 #1000 iters can be increased for more accuracy ,just for debugging.
    params,eror=model(x,y,iterss)
    p.dump(params,open('fmnistparams.d','wb'))
    plt.plot(np.arange(iterss),eror)
    plt.show()

if __name__=='__main__':
    mainfun(x,y)
