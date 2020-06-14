import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import lib_distributions
from lib_distributions import *


card_X = 200
mean_X = 0
var_X = 2
card_Y = 300
card_Z = 4
SNRdb = 4
x_lim = 10
y_lim = 10
normal_distributions = lib_normDist( card_X, mean_X, var_X, card_Y, card_Z, SNRdb, x_lim , y_lim)
a = lib_cal_true_scalar_variance(normal_distributions.x_amps,normal_distributions.p_x,mean_X)

DKL_difference = normal_distributions.actualDKL - normal_distributions.analytic_DKL

z = np.arange(1,5)
y= normal_distributions.y_amps
X,Y = np.meshgrid(z,y)
Z = DKL_difference
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cstride=1,cmap='viridis', edgecolor='none')
ax.set_xlabel('z')
ax.set_ylabel('y')
ax.set_zlabel('KLD_Difference')

plt.figure(2)
plt.plot(normal_distributions.p_x_given_z[:,0],label="z=1")
plt.plot(normal_distributions.p_x_given_z[:,1],label="z=2")
plt.plot(normal_distributions.p_x_given_z[:,2], label="z=3")
plt.plot(normal_distributions.p_x_given_z[:,3], label="z=4")
plt.xlabel('Amplitudes')
plt.ylabel('pmf(x)')
plt.title('pmf p(x|z) |z|=%i'%normal_distributions.card_Z)


