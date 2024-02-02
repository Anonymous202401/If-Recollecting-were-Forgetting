import random
import time
import matplotlib
matplotlib.use('Agg')
from utils.options import args_parser

from Evaluate_Euclidean import Evaluate_Euclidean
from Evaluate_Euclidean import Evaluate_EuclideanTEST

from Evaluate_CKA import Evaluate_CKA
from Evaluate_CKA import Evaluate_CKATest


args = args_parser()
Evaluate_EuclideanTEST(args)
Evaluate_Euclidean(args)
Evaluate_CKA(args)
