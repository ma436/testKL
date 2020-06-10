import numpy as np
from scipy.stats import norm

from lib_realDistributions import *
from analyticalApprox import *

class quantizer_dist:
    def __init__(self, card_Z, card_Y):
        self.card_Z = card_Z
        self.card_Y = card_Y
        N_ones = np.floor(card_Y / card_Z)
        p_tmp = np.zeros([card_Z, card_Y])
        for run in range(0, card_Z):
            ptr = range(run * int(N_ones), min(((run + 1) * int(N_ones)), card_Y))
            p_tmp[run, ptr] = 1
        if ptr[-1] < card_Y:
            p_tmp[run, range(ptr.stop, np.size(p_tmp, 1))] = 1
        self.p_z_given_y = p_tmp.transpose(1,0)

class lib_normDist:
    def __init__(self, card_X, mu_x, var_X, card_Y, card_Z, SNRdb, x_lim , y_lim):
        self.card_X = card_X
        self.card_Y = card_Y
        self.card_Z = card_Z

        self.SNR = 10 ** (SNRdb / 10)
        x_var = var_X
        n_var = x_var / self.SNR
        x_lim = x_lim
        y_lim = y_lim

        self.x_amps = np.linspace(-x_lim, x_lim, self.card_X)

        self.p_x = norm.pdf(self.x_amps, loc=mu_x, scale=np.sqrt(x_var))
        self.p_x = self.p_x / self.p_x.sum()

        self.y_amps = np.linspace(-y_lim, y_lim, self.card_Y)


        self.p_y_given_x = np.zeros((card_X, card_Y))
        for i in range(card_X):
            self.p_y_given_x[i, :] = norm.pdf(self.y_amps, loc=self.x_amps[i], scale=np.sqrt(n_var))
            self.p_y_given_x[i, :] = self.p_y_given_x[i, :] / ((self.p_y_given_x[i, :]).sum())

        # Calculate p(x,y)
        p_x_mat = np.repeat(self.p_x[:, np.newaxis], card_Y, axis=1)
        self.p_y_and_x = self.p_y_given_x * p_x_mat
        # Calculate p(y)
        self.p_y = self.p_y_and_x.sum(axis=0)
        # Calculate p(x|y)
        self.p_x_given_y = (np.repeat(self.p_y[:, np.newaxis], card_X, axis = 1)).transpose(1,0)
        self.p_x_given_y = np.nan_to_num((self.p_y_and_x / self.p_x_given_y), nan=0, posinf=0)

        # Calculate mean and variance of dist p(x|y)
        self.mu_x_given_y = lib_calc_mu_x_given_y(self.x_amps, self.p_x_given_y)
        self.var_x_given_y = lib_cal_var_x_given_y(self.x_amps,self.mu_x_given_y, self.p_x_given_y)

        # Get Uniform quantizer distribution p(z|y)
        uq0 = quantizer_dist(card_Z, card_Y)

        # Calculate p(x,y,z)
        p_y0_mat = (np.repeat(self.p_y[:, np.newaxis], card_X, axis=1)).transpose(1,0)
        p_y0_mat = (np.repeat(p_y0_mat[:,:, np.newaxis], card_Z, axis=2))
        self.p_x_given_y_mat = (np.repeat(self.p_x_given_y[:,:, np.newaxis], card_Z, axis=2))
        p_z_given_y_mat = ((np.repeat(uq0.p_z_given_y[:,:, np.newaxis], card_X, axis=2)).transpose(0,2,1)).transpose(1,0,2)
        self.p_xyz = p_y0_mat * self.p_x_given_y_mat * p_z_given_y_mat

        # Calculate p(x,z)
        p_x_and_z = self.p_xyz.sum(axis = 1)

        # Calculate p(z)
        self.p_z = p_x_and_z.sum(axis=0)

        # Calculate p(x|z)
        p_x_given_z = (np.repeat(self.p_z[:, np.newaxis], card_X, axis=1)).transpose(1,0)
        self.p_x_given_z = np.nan_to_num((p_x_and_z / p_x_given_z), nan=0, posinf=0)
        self.p_x_given_z_mat = ((np.repeat(self.p_x_given_z[:,:, np.newaxis], card_Y, axis=2)).transpose(0,2,1))

        # Calculate DKL(P(X|Y)||P(X|Z)
        self.actualDKL = lib_cal_DKL_mat(self.p_x_given_y_mat,self.p_x_given_z_mat)


        # Now we verify that the analytical expression of DKL is approximately close to the true DKL calculated above
        # We use exact mean and variance so we are sure that the means and variances are not assumed but rather same as from our distributions

        #Calculate p(y|z) and mean(y|z)
        p_y_and_z = np.sum(self.p_xyz, axis =0)
        p_z_mat_y = (np.repeat(self.p_z[:,np.newaxis], card_Y, axis = 1)).transpose(1,0)
        p_y_given_z = np.nan_to_num((p_y_and_z / p_z_mat_y), nan=0, posinf=0)
        self.mu_y_given_z = lib_calc_mu_x_given_y(self.y_amps, p_y_given_z)
        
        # Calculate mu_x|z
        self.mu_x_given_Z = (self.SNR/(1+self.SNR))*self.mu_y_given_z + (mu_x/(1+self.SNR))

        # Calculate my_x|y
        # Since y is continuous mu_x|y is also a continuous variable. We will calculate means for only y_vec
        self.mu_x_given_y = (self.SNR/(1+self.SNR))*self.y_amps + (mu_x/(1+self.SNR))

        # Calculate var_x|y
        self.var_x_given_y = (x_var* n_var)/(x_var+n_var)

        # Calculate var_x|z
        rep_mu_x_given_y_sq = (np.repeat(self.mu_x_given_y[:,np.newaxis], card_Z, axis = 1))**2
        rep_mu_x_given_z_sq = ((np.repeat(self.mu_y_given_z[:,np.newaxis], card_Y, axis = 1)).transpose(1,0))**2
        self.var_x_given_z = (self.var_x_given_y  + rep_mu_x_given_y_sq)-rep_mu_x_given_z_sq

        # Calculate DKL

        rep_var_x_given_y = np.repeat(self.var_x_given_y, card_Y, axis=0)
        rep_var_x_given_y = np.repeat(rep_var_x_given_y[:,np.newaxis], card_Z, axis=1)

        rep_var_x_given_z = np.repeat(self.var_x_given_z[:,np.newaxis], card_Z, axis = 1)
        self.analytic_DKL = lib_cal_DKL_gauss(rep_var_x_given_y, self.var_x_given_z , rep_mu_x_given_y_sq, rep_mu_x_given_z_sq)
        






