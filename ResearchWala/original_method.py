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

zeros_set = tuple( [0] + list(np.exp( 1j * np.linspace(0,2*np.pi,10, endpoint=False)) ) )
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
s =math.ceil( 0.26632 * math.log(d) )
N =math.ceil( 8.32547 * d * math.log(d) )

x0s = []
for v in range(1,s+1):
    r = (1 + 2**0.5) * ((d-1)/d)**((2*v-1)/(4*s))
    for j in range(N):
        theta = 2 * math.pi * j / N + (v%2) * math.pi / N
        x0s.append( r*np.exp(1j * theta) )

#defining some functions
pprime = np.polyder(p)
fun = lambda x: np.polyval(p,x)
funprime = lambda x: np.polyval(pprime,x)

zeros = []
for x0 in x0s:
    try:
        zero = newton(func=fun, x0=x0, fprime=funprime, maxiter=N)
    except RuntimeError:
        continue
    zeros.append(zero)


# filtering out zeros by evaluating on function
zeros = np.array([z * upper_bound for z in zeros if np.isclose(abs(fun(z)),0,atol=1e-4)])



if False:
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
else:
    zeros = list(zeros)
    roots = [zeros[-1]]
    for i in range(0,len(zeros)-1):
        flag = 0
        for j in range(i+1,len(zeros)):
            if np.isclose(zeros[i],zeros[j],atol=1e-4):
                flag = 1
        if flag == 0:
            roots.append(zeros[i])

    print('Number of roots after eliminating duplicates:',len(roots))

# print('Displaying with 4 digits of precision.')
# for r in roots:
    # print(np.round(r,4))


def rad_to_index(r):
    return round(np.angle(r) / (2*np.pi) * (d-1),4)

root_count = 0
roots.sort(key=rad_to_index) # easier to detect errors
for r in roots:
    flag = 0
    for z in zeros_set:
        if np.isclose(r,z,atol=1e-4):
            flag = 1
    if flag:
        root_count += 1
        print('Root detected!')
        print(round(np.abs(r),4), rad_to_index(r))
    else:
        print('Mismatch detected!')
        print(round(np.abs(r),4),rad_to_index(r) , round(np.abs(fun(r)),4),rad_to_index(fun(r)))

if root_count != d:
    print('Number of roots found:',root_count)
    print('ERROR!')
else:
    print('All clear!')
