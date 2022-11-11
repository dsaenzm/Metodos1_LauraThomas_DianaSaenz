# TALLER 8 - Probabilidad

import numpy as np
import matplotlib.pyplot as plt


# Generales de probabilidad


# 8)  Experimento virtual

coin = np.array([1, -1])

N = 10**5
coins = 4

Lcoin = np.zeros((N,4))
exitos = 0
for i in range(N):
    for j in range(coins):
        Lcoin[i][j] = np.random.choice(coin)
        
    if Lcoin[i].sum() == 0:
        exitos += 1

P = exitos/N

print(P)

fig = plt.figure(figsize =(6, 5))
b = ['p', 'q']
H = [P, 1-P]
plt.bar(b, H, width=0.4)
plt.ylim(0,1)
plt.xlim(-0.8,1.8)
plt.title("Probabilidad de obtener 2 caras y 2 sellos (p)")
for i in range(len(b)):
    plt.text( b[i], 0.02 + np.round( H[i],2 ), \
            str(np.round( H[i],3 )), ha='center',fontsize=15 )

