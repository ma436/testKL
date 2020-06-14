import numpy as np

def lib_cal_DKL_gauss(var_p, var_q, mu_p, mu_q):
    var_ratio = var_p/ var_q
    mean_ratio = (mu_p-mu_q)**2/var_q
    log_ratio = np.nan_to_num(np.log2(var_ratio))

    DKL_Gauss = (var_ratio + mean_ratio - 1 + log_ratio)/2
    return DKL_Gauss

def lib_cal_derived_var_x_given_z(var_x_given_y, p_y_given_z, mu_x_given_y, mu_x_given_z):
    card_Y, card_Z = np.shape(p_y_given_z)
    rep_var_x_given_y = np.repeat(var_x_given_y[:, np.newaxis], card_Z, axis=1)
    rep_mu_x_given_y = np.repeat(mu_x_given_y[:, np.newaxis], card_Z, axis=1)
    rep_mu_x_given_z = (np.repeat(mu_x_given_z[:,np.newaxis], card_Y, axis=1)).transpose(1,0)

    var_x_given_z_term1 = (rep_var_x_given_y*p_y_given_z).sum(axis=0)
    var_x_given_z_term2 = ((rep_mu_x_given_y-rep_mu_x_given_z)**2 * p_y_given_z).sum(axis=0)
    var_x_given_z = var_x_given_z_term1 + var_x_given_z_term2
    return var_x_given_z
