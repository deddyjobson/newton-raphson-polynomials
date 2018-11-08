import numpy as np
import argparse
import math
import multiprocessing
import png
import warnings
import os

# from scipy.optimize import newton
from sklearn.cluster import KMeans
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--resolution',type=int,default=1024)
parser.add_argument('--extent',type=float,default=2)
parser.add_argument('--maxiter',type=int,default=100)
parser.add_argument('--tol',type=float,default=1e-8)
parser.add_argument('--save_file',type=str,default='')

params = parser.parse_args()

zeros = (0,0,0,1)
p = np.poly( zeros )
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

funprime = lambda x: np.polyval(pprime,x)

z0shape = z0s.shape
z0s = z0s.reshape(-1)
nmap = np.zeros(z0s.shape, dtype=np.complex_)

def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None):
    """
    Find a zero using the Newton-Raphson or secant method.
    Find a zero of the function `func` given a nearby starting point `x0`.
    The Newton-Raphson method is used if the derivative `fprime` of `func`
    is provided, otherwise the secant method is used.  If the second order
    derivative `fprime2` of `func` is provided, then Halley's method is used.
    Parameters
    ----------
    func : function
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    fprime : function, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value.
    maxiter : int, optional
        Maximum number of iterations.
    fprime2 : function, optional
        The second order derivative of the function when available and
        convenient. If it is None (default), then the normal Newton-Raphson
        or the secant method is used. If it is not None, then Halley's method
        is used.
    Returns
    -------
    zero : float
        Estimated location where function is zero.
    """
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    # Multiply by 1.0 to convert to floating point.  We don't use float(x0)
    # so it still works if x0 is complex.
    p0 = 1.0 * x0
    plist = [p0]


    # Newton-Rapheson method
    for iter in range(maxiter):
        fder = fprime(p0, *args)
        if fder == 0:
            msg = "derivative was zero."
            warnings.warn(msg, RuntimeWarning)
            return p0,0,0 # give some dummy value
        fval = func(p0, *args)
        newton_step = fval / fder
        p = p0 - newton_step
        plist.append(p)
        if abs(p - p0) < tol:
            val = any([math.isclose(abs(pt-2/3),0,abs_tol=1e-1) for pt in plist])
            return p, iter, val # returning number of iterations also, and pre-image of root
        p_1 = p0
        p0 = p

    msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
    raise RuntimeError(msg)

darkness = np.empty_like(nmap,dtype = np.int_)
vals = np.empty_like(darkness,dtype = np.int_)
for i in range(len(nmap)):
    try:
        nmap[i],darkness[i],vals[i] = newton(func=fun, x0=z0s[i], fprime=funprime, tol=params.tol, maxiter=params.maxiter)
    except RuntimeError:
        nmap[i],darkness[i],vals[i] = 2, params.maxiter, 0 # all roots lie within unit circle, so this should work

# zeros = list(enumerate(zeros))
img = np.zeros(shape=(params.resolution,params.resolution,3),dtype=np.uint8)

colors = [np.array((255,0,0)),np.array((0,255,0)),np.array((0,0,255))] # sufficiently spaced apart
black = np.array((0,0,0))

nmap = nmap.reshape(z0shape)
darkness = darkness.reshape(z0shape)
vals = vals.reshape(z0shape)


for x in range(100):
    # creating image
    for i in range(len(nmap[0])):
        for j in range(len(nmap)):
            root = nmap[i][j] * upper_bound
            if math.isclose(abs(root-0),0,abs_tol=1e-4) and darkness[i][j]<=x:
                img[i][j] = colors[darkness[i][j] % 3]
            else:
                img[i][j] = black
    png.from_array(img, 'RGB').save(os.path.join('CumulativeIter',"{0}_iter.png".format(x)))
    # png.from_array(img, 'RGB').save("NewtonMapDiscrete{0}.png".format(params.resolution))
