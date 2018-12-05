import numpy as np
import argparse
import math
import pickle

from scipy.optimize import newton
from sklearn.cluster import KMeans
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--save_file',type=str,default='')
parser.add_argument('--tol',type=float,default=1e-12)

params = parser.parse_args()

zeros_set = tuple( [0] + list(np.exp( 1j * np.linspace(0,2*np.pi,100, endpoint=False)) ) )
p = np.poly( zeros_set )
# p = np.poly( (1j,2+3j,3-1j,4,5+5j) )
d = len(p) - 1

p *= 0
p[-1] = 0
p[-2] = -1
p[0] = 1
# print(p)
# exit()

# print('Polynomial:',p)


# obtaining upper bound on roots using Cauchy
if True:
    p /= p[0] # to make it monic
    coeffs = list(np.absolute(p))[1:]
    upper_bound = max(map( lambda x:x[1]**(1/(x[0]+1)) , enumerate(coeffs) ))
    p /= upper_bound**np.arange(p.shape[0])
else:
    upper_bound = 1


# print('Polynomial:',p)
# exit()

#obtaining starting points
s =math.ceil( 0.26632 * math.log(d) )
N =math.ceil( 8.32547 * d * math.log(d) )

x0s = []
for v in range(1,s+1):
    r = (1 + math.sqrt(2)) * ((d-1)/d)**((2*v-1)/(4*s))
    for j in range(N):
        theta = 2 * math.pi * j / N + (v%2) * math.pi / N
        x0s.append( r*np.exp(1j * theta) )

#defining some functions
pprime = np.polyder(p)
fun = lambda x: np.polyval(p,x)
funprime = lambda x: np.polyval(pprime,x)

try: # for now to save time
    zeros = pickle.load(open('zeros100.pkl','rb'))
except FileNotFoundError:
    zeros = []
    for x0 in x0s:
        try:
            zero = newton(func=fun, x0=x0, fprime=funprime, maxiter=s*N, tol=params.tol)
        except RuntimeError:
            print('Starting point {0} failed to converge.'.format(x0))
            continue
        zeros.append(zero)
    pickle.dump(zeros, open('zeros100.pkl','wb'))


# filtering out zeros by evaluating on function
print('Number of roots before validating if root:',len(zeros))
zeros = np.array([z * upper_bound for z in zeros if np.isclose(abs(fun(z)),0,atol=math.sqrt(params.tol))])

def fun2(z):
    return upper_bound * fun(z/upper_bound)

def np2(z):
    return z - fun(z/upper_bound) / funprime(z/upper_bound)


if False:
    roots = list(zeros)
else:
    zeros = list(zeros)
    roots = [zeros[-1]]
    print('Number of roots before eliminating duplicates:',len(zeros))
    for i in range(0,len(zeros)-1):
        if not np.any( np.isclose(zeros[i],zeros[i+1:],atol=math.sqrt(params.tol)) ):
            roots.append(zeros[i])

    print('Number of roots after eliminating duplicates:',len(roots))

# print('Displaying with 4 digits of precision.')
# for r in roots:
    # print(np.round(r,4))


def formatted(r):
    return round(np.abs(r),4), round(np.angle(r) / (2*np.pi) * (d-1),4)

def rad_to_index(r):
    return round(np.angle(r) / (2*np.pi) * (d-1),4)

root_count = 0
roots.sort(key=rad_to_index) # easier to detect errors
for r in roots:
    if np.any( np.isclose( r,zeros_set,atol=params.tol**(1/3) ) ):
        root_count += 1
        print('Root!')
        print(*formatted(r))
    else:
        print('Mismatch!')
        print(*formatted(r) ,'\t', *formatted(fun2(r)))

if root_count != d:
    print('Number of roots found:',root_count)
else:
    print('All clear!')
