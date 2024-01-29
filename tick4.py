import numpy as np
import matplotlib.pyplot as plt


def pairs(N):
    permvec = np.random.permutation(N)
    return (permvec[:N//2],permvec[N//2:])
    
def kinetic_exchange(v, w):
    R = np.random.uniform(size = v.size)    
    total = v + w
    vnew = R * total
    wnew = (1 - R) * total
    return (vnew, wnew)

def gini(w):
    N = w.size
    sorted_vector = np.sort(w)
    indexes = np.arange(N) + 1
    numerator = np.sum(sorted_vector * indexes)
    denomenator = N * np.sum(w)
    return (2 * (numerator/denomenator)) - (1 + 1/N)

def sim(N,T,salary,roc):
    w = np.ones(N)
    gs = []
    for i in range(T-1):
        w = w + salary
        w = w * (1+roc)
        gs.append(gini(w))
        (x,y) = pairs(N)
        (vnew,wnew) = kinetic_exchange(w[x],w[y])
        wtemp = np.concatenate((vnew,wnew))
        w = wtemp[np.argsort(np.concatenate((x,y)))]
        #w = normalize_vector(w)

    return (w,gs)

def normalize_vector(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    
    normalized_vector = 2 * (vector - min_val) / (max_val - min_val)
    return normalized_vector

def generate_line(N,T,roc): 
    salary = np.random.uniform(size = N)
    (_,gs) = sim(N,T,salary,roc)
    x = np.linspace(0,T-2,T-1)
    return (x,gs)


fig,ax = plt.subplots()
for i in range(0,5):
    (x,gs) = generate_line(500000,30,i/10)
    ax.plot(x, gs, linestyle='--', alpha=0.7, label = f'Return on Capital = {i/10}')
plt.legend()
ax.set_xlabel("Timestep")
ax.set_ylabel("Gini Coefficient")
plt.show()
