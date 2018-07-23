import numpy as np
import argparse

from scipy.optimize import newton


parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--save_file',type=str,default='')

params = parser.parse_args()
