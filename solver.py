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

if params.default:
    p = np.poly( (1,2,3,4,5) )
    # p = np.poly( (1j,2+3j,3-1j,4,5+5j) )
    d = len(p) - 1
else:
    choice = 0
    while choice not in (1,2):
        print('How do you want to input the polynomial?')
        print('1.Coefficients')
        print('2.Roots')
        choice = int(input())
        if choice == 1:
            d = int(input('Enter the degree of the polynomial:'))
            p = np.zeros(shape=d+1,dtype=np.complex_)
            p[0] = 1 # default value
            print('Enter the exponent and the coefficient in pairs in this format:')
            print('\nEx: For d = 4')
            print('0,1.1 3,-1 2,2-1j    ->    x^4 - x^3 + (2-1j)x^2 + 0x + 1.1\n')
            coeffs = input('Coefficients:').strip().split()
            coeffs = map(lambda x:x.split(',') , coeffs)
            for pos,val in coeffs:
                pos = int(pos)
                if pos > d or pos < 0:
                    exit('Illegal exponent value')
                val = complex(val)
                p[pos] = val
        elif choice == 2:
            print('Enter the zeros separated by spaces:')
            zeros = input().strip().split()
            zeros = list(map(complex,zeros))
            p = np.poly(zeros)
            d = len(p) - 1
        else:
            pass


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
    zero = newton(func=fun, x0=x0, fprime=funprime)
    zeros.append(zero)

# filtering out zeros by evaluating on function
zeros = np.array([z * upper_bound for z in zeros if math.isclose(abs(fun(z)),0,abs_tol=1e-4)]) # WOT!

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

print('Displaying with 4 digits of precision.')
for r in roots:
    print(np.round(r,4))
