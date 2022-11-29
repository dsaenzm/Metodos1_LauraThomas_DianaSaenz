import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations_with_replacement

#Parámetros necesarios
#Punto a
T = np.array([[0.8,0.2],[0.2,0.8]])
E = np.array([[0.5,0.9],[0.5,0.1]])
Prior = np.array([0.2,0.8]) 
State = np.array([0,1])
listt = []

#Código clase

def GetHiddenStates(States, N):
    CStates = list( combinations_with_replacement(States,N) )
    permutacion = []
    for it in CStates:
        p = list(permutations(it,N)) 
        for i in p:
            if i not in permutacion:
                permutacion.append(i)
    return np.array(permutacion)

Ocultos = GetHiddenStates(State,8)

for i in range(len(Ocultos)):
    Obs = Ocultos[i]
    def GetProb(T,E,Obs,State,Prior):
      n = len(Obs)
      p = 1
      p*= Prior[State[0]]
      for y in range(n-1):
          p *= T[ State[y+1], State[y] ]
      for y in range(n):
          p *= E[ Obs[y], State[y] ]
      return p

    dim = Ocultos.shape[0]
    P = np.zeros(dim)
    
    for y in range(dim):
        P[y] = GetProb(T,E,Obs,Ocultos[y],Prior)
    
    maxP = np.max(P)
    iy = np.where( P == np.amax(P))
    
    listt.append(Ocultos[iy][0])

    if (np.array(Ocultos[iy][0]) == np.array([1,1,1,1,0,0,0,0])).all() == True and (np.array(Obs)==np.array([1,0,0,0,1,0,1,0])).all() == True:
        
        print("Punto b: Secuencia más probable:",Ocultos[iy][0],"que tiene probabilidad de:",maxP)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(P,color='k',label='Probabilidad por secuencia')
        ax.axhline(y=maxP,c='r',label='MaxP = 0.0002')
        ax.legend(loc = 'upper left')
        
        ObsStates = GetHiddenStates([0,1],8)
        
        NObs = ObsStates.shape[0]
        
        PObs = np.zeros(NObs)
        
        for j in range(NObs):
            
            dim = Ocultos.shape[0]
            P = np.zeros(dim)
            
            for y in range(dim):
                P[y] = GetProb(T,E,ObsStates[j],Ocultos[y],Prior)
                
            PObs[j] = np.sum(P)
        
        fig2 = plt.figure(figsize=(6,6))
        ax1 = fig2.add_subplot(1,1,1)
        ax1.plot(PObs,label = 'Probabilidad de los eventos observables',c='b')
        ax1.legend(loc = 'upper right')
        
        maxP = np.max(PObs)
        jj = np.where( PObs == np.amax(PObs))
        
        #Punto d
        print("Para el b: El estado más probable:",ObsStates[jj][0],"tiene probabilidad de:",round(maxP,3))
        
        print("Para el punto d. La suma de todas las probabilidades de los estados observables es:",np.sum(PObs))
        
def most_repeted_row(Matrix):
    Matrix = np.ascontiguousarray(Matrix)
    void_dt = np.dtype((np.void,Matrix.dtype.itemsize*np.prod(Matrix.shape[1:])))
    _,ids,count = np.unique(Matrix.view(void_dt).ravel(),return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = Matrix[largest_count_id]
    return most_frequent_row,len(ids)

print("Las secuencia oculta más probable es: "+str(most_repeted_row(listt)[0])+"Es más probable en"+str(most_repeted_row(listt)[1])+"por combinaciones distintas da aproximadamente un 7,8. Por otro lado, por esto, para el punto c la respuesta es no")