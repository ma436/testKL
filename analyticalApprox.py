import numpy as np

def lib_cal_DKL_gauss(var_p, var_q, mu_p, mu_q):
    var_ratio = var_p/ var_q
    mean_ratio = (mu_p-mu_q)**2/var_q
    log_ratio = np.nan_to_num(np.log2(var_ratio))

    DKL_Gauss = (var_ratio + mean_ratio - 1 + log_ratio)/2
    return DKL_Gauss