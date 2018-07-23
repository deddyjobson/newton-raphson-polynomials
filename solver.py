import numpy as np
import argparse
import math

from scipy.optimize import newton
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--display',type=int,default=0)
parser.add_argument('--save_file',type=str,default='')

params = parser.parse_args()

if params.default:
    p = np.poly( (1,2,3,4,5) )
    # p = np.poly( (1j,2+3j,3-1j,4,5+5j) )
    d = len(p) - 1
else:
    pass


# obtaining upper bound on roots using Cauchy
coeffs = list(np.absolute(p))[1:]
upper_bound = max(map( lambda x:x[1]**(1/(x[0]+1)) , enumerate(coeffs) ))
p /= upper_bound**np.arange(p.shape[0])
p /= p[0] # to make it monic


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

zeros = np.array([z * upper_bound for z in zeros])


# now to extract the zeros...
zeros = zeros.reshape((-1,1))
kmeans = KMeans(n_clusters=d).fit(zeros)
# print(*list(zip( zeros,kmeans.predict(zeros) )) , sep='\n')

labels = kmeans.predict(zeros)
zeros = zeros.reshape(-1)


#reorganizing all clusters
roots = [[] for _ in range(d)]
for i,z in enumerate(zeros):
    roots[labels[i]].append(z)

for i,r in enumerate(roots):
    roots[i] = sum(r)/len(r)

print('Displaying with 4 digits of precision.')
for r in roots:
    print(np.round(r,4))

# to display newton fractals
if params.display:
    # Newton fractals
    # Based on FB - 201003291, http://code.activestate.com/
    from PIL import Image
    import matplotlib.pyplot as plt

    imgx = 800
    imgy = 800
    image = Image.new("RGB", (imgx, imgy))

    # Complex window
    xa = -1.0
    xb = 1.0
    ya = -1.0
    yb = 1.0

    maxIt = 20 # max iterations allowed
    h = 1e-6 # step size for numerical derivative
    eps = 1e-3 # max error allowed

    # put any complex function here to generate a fractal for it!
    def f(z):
        return z * z * z - 1.0

    # draw the fractal
    print("Running complex newton iterations from %i x %i pixels" % (imgx,imgy), end="... ")
    for y in range(imgy):
        zy = y * (yb - ya) / (imgy - 1) + ya
        for x in range(imgx):
            zx = x * (xb - xa) / (imgx - 1) + xa
            z = complex(zx, zy)
            for i in range(maxIt):
                # approximation of complex numerical derivative
                dz = (f(z + complex(h, h)) - f(z)) / complex(h, h)
                z0 = z - f(z) / dz # Newton iteration
                if abs(z0 - z) < eps: # stop when close enough to any root
                    break
                z = z0
            r, g, b = i % 4 * 64, i % 8 * 32, i % 16 * 16 # red, green, blue
            image.putpixel((x, y), (r,g,b))
    print ("¡bien!")












pass
