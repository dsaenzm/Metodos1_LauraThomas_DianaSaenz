import numpy as np
import matplotlib.pyplot as plt

tan = lambda x: np.sqrt(np.tan(x))
x = np.linspace(0.1,1.1)
f = tan(x)
h = 0.01

def DerivadaProgresiva(f,x,h):
    
    a  =((-3*f(x)) + (4*f(x+h)) - (f(x+(2*h))))
    return (1/(2*h))*a

c = DerivadaProgresiva(tan, x, h)
print(c)
print(plt.plot(c))

def DerivadaCentral(f,x,h):
    a = f(x+h) - f(x-h) / (2*h)
    return a

d = DerivadaCentral(tan, x, h)
print(d)
print(plt.plot(d))
