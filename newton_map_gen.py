import numpy as np
import argparse
import math
import multiprocessing
import png

from scipy.optimize import newton
from sklearn.cluster import KMeans
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--resolution',type=int,default=64)
parser.add_argument('--extent',type=float,default=2)
parser.add_argument('--save_file',type=str,default='')

params = parser.parse_args()

if params.default:
    zeros = (1,2,3,4,5)
    zeros = (1j,2+3j,3-1j,4,5+5j)
    p = np.poly( zeros )
    d = len(p) - 1
else:
    print('Enter the zeros separated by spaces:')
    zeros = input().strip().split()
    zeros = list(map(complex,zeros))
    p = np.poly(zeros)
    d = len(p) - 1



# obtaining upper bound on roots using Cauchy
p /= p[0] # to make it monic
coeffs = list(np.absolute(p))[1:]
upper_bound = max(map( lambda x:x[1]**(1/(x[0]+1)) , enumerate(coeffs) ))
p /= upper_bound**np.arange(p.shape[0])
p /= p[0] # to ensure it's monic


xx,yy = np.meshgrid(np.linspace(-1*params.extent,params.extent,params.resolution), np.linspace(-1*params.extent,params.extent,params.resolution))
z0s = xx+1j*yy


#defining some functions
pprime = np.polyder(p)
fun = lambda x: np.polyval(p,x)

def fun1(x):
    print(x)
    return np.polyval(p,x)

funprime = lambda x: np.polyval(pprime,x)

z0shape = z0s.shape
z0s = z0s.reshape(-1)
nmap = np.zeros(z0s.shape, dtype=np.complex_)

for i in range(len(nmap)):
    try:
        nmap[i] = newton(func=fun, x0=z0s[i], fprime=funprime)
    except RuntimeError:
        nmap[i] = 2 # all roots lie within unit circle, so this should work

# zeros = list(enumerate(zeros))
img = np.zeros(shape=(params.resolution,params.resolution,3),dtype=np.uint8)
colors = [np.random.randint(0,255,size=3,dtype=np.uint8) for _ in zeros]


nmap = nmap.reshape(z0shape)

# creating image
for i in range(len(nmap[0])):
    for j in range(len(nmap)):
        root = nmap[i][j] * upper_bound
        for z,zero in enumerate(zeros):
            if math.isclose(abs(root-zero),0,abs_tol=1e-4):
                img[i][j] = colors[z]
                break


png.from_array(img, 'RGB').save("NewtonMap{0}.png".format(params.resolution))
