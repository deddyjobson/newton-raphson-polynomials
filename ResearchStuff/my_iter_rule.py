import numpy as np
import argparse
import math
from mpmath import mp
mp.dps = 100
mp.pretty = True

from scipy.optimize import newton
from sklearn.cluster import KMeans
from scipy.special import comb
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--save_file',type=str,default='')

params = parser.parse_args()


# p = np.poly( (1,2,3,4,5) )
# zeros_set = tuple( [0] + list(np.exp( 1j * np.linspace(0,2*np.pi,50, endpoint=False)) ) )

def msum_zeros_to_poly(zeros):
    poly_size = len(zeros) + 1
    poly = np.zeros(poly_size, dtype=np.complex_)
    poly[0] = 1

    for zero in zeros:
        partials = []
        i = 0
        # print(x)
        x = np.roll(-zero*poly , 1)
        # print(x)
        # exit()
        for y in partials:
            s1 = abs(x) < abs(y)
            # s2 = (1-s1).astype(np.bool)
            x[s1], y[s1] = y[s1], x[s1]
            assert np.all(abs(x)>=abs(y)) # just making sure
            hi = x + y
            lo = y - (hi - x)
            if True:
                partials[i] = lo
                i += 1
            x = hi
        partials[i:] = [x]

        partials = np.array(partials)
        poly += np.sum(partials, axis=0)
        # print(poly)
    return poly

def kahan_zeros_to_poly(zeros):
    poly_size = len(zeros) + 1
    poly = np.ones(1)

    c = np.zeros(poly_size, dtype=np.complex_) # compensation

    for zero in zeros:
        y = - zero * poly - c[:poly.shape[0]]
        y = np.pad(y, (1, 0), 'constant')

        t = np.pad(poly, (0, 1), 'constant') + y
        c[:poly.shape[0]+1] = ( t - np.pad(poly, (0, 1), 'constant') ) - y
        poly = t
    return poly

def precise_kahan_zeros_to_poly(zeros):
    poly_size = len(zeros) + 1
    poly = np.array([mp.mpc(1) for _ in range(1)], dtype=np.object)

    c = np.array([mp.mpc(0) for _ in range(poly_size)], dtype=np.object)

    for zero in zeros:
        y = - zero * poly - c[:poly.shape[0]]
        y = np.pad(y, (1, 0), 'constant', constant_values=mp.mpc(0))

        t = np.pad(poly, (0, 1), 'constant', constant_values=mp.mpc(0)) + y
        c[:poly.shape[0]+1] = ( t - np.pad(poly, (0, 1), 'constant', constant_values=mp.mpc(0)) ) - y
        poly = t
    return poly

def neumaier_zeros_to_poly(zeros):
    poly_size = len(zeros) + 1
    poly = np.ones(1)

    c = np.zeros(poly_size, dtype=np.complex_) # compensation

    for zero in zeros:
        new_item = np.pad( - zero * poly , (1, 0), 'constant')
        poly = np.pad(poly, (0, 1), 'constant')

        t = poly + new_item
        s1 = abs(poly) > abs(new_item)
        s2 = (1 - s1).astype(np.bool)
        c[:poly.shape[0]] = ( t - poly*s1 - new_item*s2 ) - new_item*s1 - poly*s2
        poly = t
    return poly

def divide_conquer_zeros_to_poly(zeros):
    terms = [np.array([1,-x], dtype=np.complex_) for x in zeros]
    while len(terms) > 1:
        temp = [np.convolve(terms[i],terms[i+1]) for i in range(0,len(terms)-1,2)]
        if len(terms) % 2 != 0:
            terms = temp + [terms[-1]]
        else:
            terms = temp
    return terms[0]


if False:
    NUM_ROOTS = 50
    zeros_set = tuple( list(np.sqrt(np.random.uniform(0,1,NUM_ROOTS)) * np.exp( 1j * np.random.uniform(0,2*np.pi,NUM_ROOTS) ) ) )
    p = np.poly( zeros_set )
else:
    # zeros_set =  [0] + list(np.exp( 1j * np.linspace(0,2*np.pi,100, endpoint=False)))
    zeros_set =  np.exp( 1j * np.linspace(0,2*np.pi,100, endpoint=False))
    np.random.shuffle(zeros_set)
    p = precise_kahan_zeros_to_poly( zeros_set )

p = np.array(p, dtype=np.complex_)

# p = np.poly( (1j,2+3j,3-1j,4,5+5j) )
d = len(p) - 1

# print(p[d//2].astype(np.complex64))
print('Maximum Error:', max(abs(p[3:-3])))

zeros_set = np.angle(zeros_set)
zeros_sorted = sorted(zeros_set)

print(zeros_set)
print(zeros_sorted)

# coeff = np.correlate(zeros_sorted,zeros_set) / np.linalg.norm(zeros_sorted) / np.linalg.norm(zeros_set)
coeff = np.corrcoef(zeros_sorted,zeros_set)
print('Correlation with sorted by argument:', abs(coeff)[0])
exit()


xx = np.arange(p.shape[0]).astype(np.int)
yy1 = abs(p)
yy2 = comb(p.shape[0],xx) * 1e-15
from pylab import *
figure(dpi=300)
title('Error profile')
semilogy(xx,yy1,label='coefficient')
semilogy(xx,yy2,label='# terms summed')
legend()
show()

exit()


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
for r in roots:
    print(np.round(r,4), np.round(np.angle(r) * 100 / (2*np.pi), 2) )

another_flag = 0
for r in roots:
    flag = 0
    if np.any( np.isclose(r,zeros_set,atol=1e-4) ):
        flag = 1
    else:
        print('Mismatch detected!')
        another_flag = 1

if another_flag:
    print('ERROR!')
else:
    print('All clear!')
