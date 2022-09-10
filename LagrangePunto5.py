import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path as path
import wget

file='InterpolacionNewtonNoequi.csv'
url='https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/InterpolacionNewtonNoequi.csv'
if not path.exists(file):
    Path_ = wget.download(url,file)
else:
    Path_ = file

Data = pd.read_csv(Path_,sep=',')
X = np.float64(Data['X'])
Y = np.float64(Data['Y'])
Diff = np.zeros((len(X),len(Y)))

def NewtonGregory(X,Y,x):
    
    Sum_ = Y[0]
    
    Diff = np.zeros((len(X),len(Y)))
    Diff[:,0] = Y


            
    poly = 1.0
    
    for i in range(1,len(X)):
        
        poly *= ( x - X[i-1] )
        
        for j in range(i, len(X)):
            
            h = X[j] -X[j-i]
            
            Diff[j,i] = (Diff[j,i-1]-Diff[j-1,i-1])/ (h)
            
            
            
        Sum_ += poly*(Diff[i,i])
    
    return Sum_,np.round(Diff,2)

x = np.linspace(np.min(X),np.max(X),100)
y,_ = NewtonGregory(X,Y,x)

plt.scatter(X,Y)
plt.plot(x,y)






