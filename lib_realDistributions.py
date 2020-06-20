import numpy as np
def lib_calc_mu_x_given_y(x_vec, p_x_given_y):
    """
     The mean is defined as mean = Sum(X.p(X|Y)
    :param x_vec:
    :param p_x_given_y:
    :return:
    """
    size_X,size_Y = np.shape(p_x_given_y)
    mu_x_given_y = np.repeat(x_vec[:,np.newaxis], size_Y, axis=1)
    mu_x_given_y = np.sum((mu_x_given_y * p_x_given_y),axis = 0)
    return mu_x_given_y

def lib_cal_var_x_given_y (x_vec,mu_x_given_y, p_x_given_y):
    """

    :param x_vec:
    :param mu_x_given_y:
    :param p_x_given_y:
    :return:
    """
    size_X, size_Y = np.shape(p_x_given_y)
    x_vec_mat = np.repeat(x_vec[:,np.newaxis], size_Y, axis=1)
    mu_x_given_y_mat = (np.repeat(mu_x_given_y[:,np.newaxis], size_X, axis=1)).transpose(1,0)
    var_x_given_y = ((x_vec_mat - mu_x_given_y_mat)**2 * p_x_given_y).sum(axis=0)

    return var_x_given_y

def lib_cal_DKL_mat(p_x_given_y,p_x_given_z):

    """
    :param p_x_given_y:
    :param p_x_given_z:
    :return:
    """
    DKL = np.sum(np.where(p_x_given_y != 0, p_x_given_y * np.log2(p_x_given_y / p_x_given_z),0),axis=0)
    return DKL

def lib_cal_true_scalar_variance(amps,pmf,mean):
    var = np.sum(((amps-mean)**2)*pmf)
    return var

def lib_KLD_long(p_x_given_y, p_x_given_t):
    Nx,Ny = np.shape(p_x_given_y)
    Nx,Nt = np.shape(p_x_given_t)
    KLD_l = np.zeros((Nt,Ny))
    for runKL in np.arange(0, Nx):
        klp1 = np.tile((np.nan_to_num(np.log2(p_x_given_t[runKL, :]))), (Ny, 1))
        klp2 = np.tile((np.nan_to_num(np.log2(p_x_given_y[runKL, :]))), (Nt, 1))
        klp12 = klp2 - np.transpose(klp1)
        klp3 = np.tile(p_x_given_y[runKL, :], (Nt, 1))
        klp123 = klp3 * klp12

        KLD_l = KLD_l + klp123
    return KLD_l.transpose(1,0)
