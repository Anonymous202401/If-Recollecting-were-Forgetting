import random
import time
import matplotlib
matplotlib.use('Agg')
from utils.options import args_parser

from Evaluate_Euclidean import Evaluate_Euclidean
from Evaluate_Euclidean import Evaluate_EuclideanTEST

args = args_parser()
Evaluate_EuclideanTEST(args)
Evaluate_Euclidean(args)
