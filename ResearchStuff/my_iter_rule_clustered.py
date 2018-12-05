import numpy as np
import argparse
import math

from scipy.optimize import newton
from sklearn.cluster import KMeans
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--save_file',type=str,default='')

params = parser.parse_args()


# p = np.poly( (1,2,3,4,5) )
# zeros_set = tuple( [0] + list(np.exp( 1j * np.linspace(0,2*np.pi,50, endpoint=False)) ) )
zeros_set = tuple( [0] + list(np.exp( 1j * np.random.uniform(0,2*np.pi,50) ) ) )
p = np.poly( zeros_set )
# p = np.poly( (1j,2+3j,3-1j,4,5+5j) )
d = len(p) - 1




# obtaining upper bound on roots using Cauchy
p /= p[0] # to make it monic
coeffs = list(np.absolute(p))[1:]
upper_bound = max(map( lambda x:x[1]**(1/(x[0]+1)) , enumerate(coeffs) ))
p /= upper_bound**np.arange(p.shape[0])
p /= p[0] # to ensure it's monic


#obtaining starting points
N =math.ceil( 2 * d * math.log(d) )

x0s = 2 * np.exp( 1j * np.linspace(0,2*np.pi,N, endpoint=False) )

#defining some functions
pprime = np.polyder(p)
fun = lambda x: np.polyval(p,x)
funprime = lambda x: np.polyval(pprime,x)

zeros = []
for x0 in x0s:
    try:
        zero = newton(func=fun, x0=x0, fprime=funprime, maxiter=N)
    except RuntimeError:
        print('Increase the limit')
        continue
    zeros.append(zero)


# filtering out zeros by evaluating on function
zeros = np.array([z * upper_bound for z in zeros if math.isclose(abs(fun(z)),0,abs_tol=1e-4)])

# now to extract the zeros...
zeros_cart = np.array([ [z.real,z.imag] for z in zeros ])
kmeans = KMeans(n_clusters=d).fit(zeros_cart)
# print(*list(zip( zeros,kmeans.predict(zeros) )) , sep='\n')

labels = kmeans.predict(zeros_cart)
del zeros_cart

#reorganizing all clusters
roots = [[] for _ in range(d)]
for i,z in enumerate(zeros):
    roots[labels[i]].append(z)

for i,r in enumerate(roots):
    roots[i] = sum(r)/len(r)

# print('Displaying with 4 digits of precision.')
# for r in roots:
    # print(np.round(r,4))

another_flag = 0
for r in roots:
    flag = 0
    for z in zeros_set:
        if np.isclose(r,z,atol=1e-4):
            flag = 1
    if not flag:
        print('Mismatch detected!')
        another_flag = 1

if another_flag:
    print('ERROR!')
else:
    print('All clear!')
