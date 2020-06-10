import numpy as np
from matplotlib import pyplot as plt
import lib_distributions
from lib_distributions import *


card_X = 200
mean_X = 0
var_X = 2
card_Y = 400
card_Z = 4
SNRdb = 4
x_lim = 10
y_lim = 10
normal_distributions = lib_normDist( card_X, mean_X, var_X, card_Y, card_Z, SNRdb, x_lim , y_lim)
a = lib_cal_true_scalar_variance(normal_distributions.x_amps,normal_distributions.p_x,mean_X)