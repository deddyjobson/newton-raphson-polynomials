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

N = 1000
d = 100

xx = np.ones((N//d, d))
yy = np.ones((N//d, d))

for amt in range(N):
    zeros_set =  np.exp( 1j * np.linspace(0,2*np.pi,d, endpoint=False))
    np.random.shuffle(zeros_set[d*amt//N:])

    p = np.poly( zeros_set )


    p = np.array(p, dtype=np.complex_)

    zeros_set = np.angle(zeros_set)
    zeros_set[zeros_set<0] += 2 * np.pi # converting to more convenient form

    zeros_sorted = sorted(zeros_set)

    coeff = np.corrcoef(zeros_sorted,zeros_set)
    
    xx[amt%(N//d),amt//(N//d)] = coeff[0,1]
    yy[amt%(N//d),amt//(N//d)] = max(abs(p[3:-3]))


from pylab import *
figure(dpi=200)
title('Error profile')
xlabel('Correlation with sorted list')
ylabel('Maximal error in coefficient')
for i in range(N//d):
    semilogy(xx[i],yy[i],'.')
show()
