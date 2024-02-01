import random
import time
import matplotlib
matplotlib.use('Agg')
from utils.options import args_parser
from Evaluate_Pearson import Evaluate_Pearson
from Evaluate_Pearson import Evaluate_PearsonTEST
from Evaluate_Euclidean import Evaluate_Euclidean
from Evaluate_Euclidean import Evaluate_EuclideanTEST
from Evaluate_Scatter import Evaluate_Scatter
from Evaluate_Scatter import Evaluate_ScatterTEST
from Evaluate_CKA import Evaluate_CKA
from Evaluate_CKA import Evaluate_CKATest


args = args_parser()
# Evaluate_ScatterTEST(args)
# Evaluate_EuclideanTEST(args)
# Evaluate_PearsonTEST(args)
# Evaluate_CKATest
Evaluate_Pearson(args)
Evaluate_Scatter(args)
Evaluate_Euclidean(args)
Evaluate_CKA(args)