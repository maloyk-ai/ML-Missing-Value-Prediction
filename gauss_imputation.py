import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from scipy.stats import multivariate_normal

from datetime import datetime as dt
import numpy as np
from functools import reduce
import time
import scipy
import scipy.io

def gaussCondition(model, x):
    mu = model['mu']
    Sigma = model['Sigma']


    o_i =  np.where(np.isnan(x) == False)[0]
    m_i =  np.where(np.isnan(x) == True)[0]
    x_o, x_m = x[o_i][:, np.newaxis], x[m_i][:, np.newaxis]

    S_MM = Sigma[np.ix_(m_i, m_i)]
    S_MO = Sigma[np.ix_(m_i, o_i)]
    S_OO = Sigma[np.ix_(o_i, o_i)]

    if len(m_i) == 3:
        return sample_gaussian(model, 1), None

    mu_m = mu[np.ix_(m_i)][:, np.newaxis] +\
                S_MO @ np.linalg.inv(S_OO) @\
                (x_o - mu[np.ix_(o_i)][:, np.newaxis])
    Sigma_m = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_MO.T
    return mu_m, Sigma_m

def sample_gaussian(model, n):
    np.random.seed(1024)
    mu = model['mu']
    Sigma = model['Sigma']
    d, _ = Sigma.shape
    X = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n)
    return X

def plot_joint_distribution(X):
    X = np.transpose(X)
    sns.jointplot(x=X[0],
                  y=X[1],
                  kind="kde",
                  space=0);
    plt.show()


def simulate_nan(nan_rate):
    mu_truth = np.array([1, 2, 6])
    Sigma_truth = np.array([[118, 62, 44], [62, 49, 17], [44, 17, 21]])
    model = { 'mu': mu_truth, 'Sigma': Sigma_truth}

    X_truth =  sample_gaussian(model, n = 2000)
    X = X_truth.copy()
    nr, nc = X.shape
    C = np.random.random((nr*nc)).reshape(nr, nc) >  nan_rate
    X[C==False] = np.nan

    result = {
        'X_truth': X_truth,
        'mu_truth': mu_truth,
        'Sigma_truth' : Sigma_truth,
        'X': X,
        'C': C,
        'nan_rate': nan_rate,
        'nan_rate_actual': np.sum(C == False) / (nr * nc)
    }

    return result

def impute_init(X, nan_rate, nan_rate_actual):
    nr, nc = X.shape
    C = np.isnan(X) == False

    # Collect M_i and O_i's
    one_to_nc = np.arange(1, nc + 1, step = 1)

    M = one_to_nc * (C == False) - 1
    O = one_to_nc * C - 1

    # Generate Mu_0 and Sigma_0
    mu = np.nanmean(X, axis = 0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]

    Sigma = np.cov(X[observed_rows, :].T)
    if np.isnan(Sigma).any():
        Sigma = np.diag(np.nanvar(X, axis = 0))

    U = np.zeros((nc, nc))
    MU = np.zeros((1, nc))
    A = X[observed_rows,: ]
    n, _ = A.shape
    for i in range(n):
        U = U + A[i,:][:, np.newaxis] @  A[i,:][:, np.newaxis].T
        MU = MU + A[i,:][np.newaxis, :] 

    np.set_printoptions(precision=3)

    MU = 1/n* MU
    print("-------------------------------------------Initialization-------------------------------------------")
    print("NaN Rate:%f" %(nan_rate))
    print("Actual NaN Rate:%f" %(nan_rate_actual))
    print("----------------------------------------------------------------------------------------------------")
    return mu, Sigma, M, O, one_to_nc, MU, U

def impute_estep(X, mu, S, M, O, one_to_nc, iteration):
    nr, nc = X.shape
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()

    for i in range(nr):
        S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)

        if set(O[i, ]) != set(one_to_nc - 1): # missing component exists
            M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1]

            S_MM = S[np.ix_(M_i, M_i)]
            S_MO = S[np.ix_(M_i, O_i)]
            S_OM = S_MO.T
            S_OO = S[np.ix_(O_i, O_i)]

            Mu_tilde[i] = mu[np.ix_(M_i)][:, np.newaxis] +\
                S_MO @ np.linalg.inv(S_OO) @\
                (X_tilde[i, O_i][:, np.newaxis]  - mu[np.ix_(O_i)][:, np.newaxis])
      
            X_tilde[i, M_i] = Mu_tilde[i].flatten()
            Vi = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM

            x_i = X_tilde[i][:, np.newaxis]
            S_i = x_i @ x_i.T
            S_i[np.ix_(M_i, M_i)] += Vi
            S_tilde[i] = S_i

    return X_tilde, S_tilde


def impute_mstep(X_tilde, S_tilde, MU, U, iteration):
    nr, nc = X_tilde.shape
    MU = np.squeeze(MU)

    mu_new =  np.mean(X_tilde, axis = 0)
    S_new = 1/nr*(U + reduce(np.add, S_tilde.values()))  - (mu_new[:, np.newaxis] @ mu_new[:, np.newaxis].T) 

    return mu_new, S_new


def impute_em(X, nan_rate, nan_rate_actual,  max_iter = 1000, eps = 1e-08):
    no_conv = True
    iterations = 0
    logprobs_hist = []

    mu, S, M, O, one_to_nc, MU, U = impute_init(X, nan_rate, nan_rate_actual)
    mu_observed = mu
    Sigma_observed = S

    while no_conv and iterations < max_iter:
        X_tilde, S_tilde = impute_estep(X, mu, S, M, O, one_to_nc, iterations)
        Mu_new, S_new = impute_mstep(X_tilde, S_tilde, MU, U, iterations)

        logprobs = compute_observed_loglikihood(Mu_new, S_new, O, one_to_nc, X)
        logprobs_hist.append(logprobs)

        no_conv = \
            np.linalg.norm(mu - Mu_new) >= eps or\
            np.linalg.norm(S - S_new, ord = 2) >= eps
        mu = Mu_new
        S  = S_new + 1e-10 * np.identity(3)
        iterations += 1

    result = {
        'X_imputed': X_tilde,
        'observed_mu': mu_observed,
        'observed_Sigma': Sigma_observed,
        'imputed_mu': mu,
        'imputed_Sigma': S,
        'iterations': iterations
    }

    return result, logprobs_hist

def compute_loglikihood(mu, Sigma, X):
    rv = multivariate_normal(mu, Sigma)
    logprobs = np.sum(np.log((rv.pdf(X)).clip(min=1e-20)), axis = 0)
    return logprobs

#Lower Bound
def compute_observed_loglikihood(mu, Sigma, O, one_to_nc, X):
    nr, _ = X.shape
    logprobs = 0
    for i in range(nr):
        Mu_tilde = mu
        S_OO = Sigma
        x_i = X[i]#[:, np.newaxis]
        if set(O[i, ]) != set(one_to_nc - 1): # missing component exists
            O_i = O[i, ][O[i, ] != -1]
            S_OO = Sigma[np.ix_(O_i, O_i)]
            Mu_tilde = mu[np.ix_(O_i)] 
            x_i = x_i[np.ix_(O_i)]
            if x_i.shape[0]== 0:
                continue
    
        rv = multivariate_normal(Mu_tilde, S_OO)
        logprobs += np.log((rv.pdf(x_i)).clip(min=1e-20))
    return logprobs


def plot_imputation(mu_truth, Sigma_truth, imputed_mu_list, imputed_Sigma_list, 
                            observed_mu_list, observed_Sigma_list, nan_rates):

    N = len(nan_rates)   

    for i in range(len(mu_truth)):
        mu_truth_i = np.repeat(mu_truth[i], N)
        Sigma_truth_ii = np.repeat(Sigma_truth[i,i], N)

        imputed_mu_i = [mu[i] for mu in imputed_mu_list] 
        imputed_Sigma_ii = [Sigma[i,i] for Sigma in imputed_Sigma_list] 

        observed_mu_i = [mu[i] for mu in observed_mu_list] 
        observed_Sigma_ii = [Sigma[i,i] for Sigma in observed_Sigma_list] 

        plt.scatter(nan_rates, mu_truth_i, s=20)
        plt.scatter(nan_rates, imputed_mu_i, s=20)
        plt.scatter(nan_rates, observed_mu_i, s=20)

        plt.legend(["Truth mu", "Imputed mu", "Observed mu"], bbox_to_anchor=(0,1.02), loc='upper left')
        plt.xlabel("NaN Rates")
        plt.ylabel("mu[%d]" %(i))
        plt.show()


def plot_observed_likelihood(figure, axes,i, j, logprobs_hist, nan_rate):
    axes[i, j].plot(np.arange(len(logprobs_hist)), logprobs_hist)
    axes[i, j].set_title("   NaN Rate: " + str(nan_rate))
    axes[i, j].set_xlabel("Iterations")
    axes[i, j].set_ylabel("Log Observed Likelihood")
    figure.canvas.draw()
    figure.canvas.flush_events()

def fill_values(X_truth, X, mu, Sigma, imputed_mu, imputed_Sigma, nan_rate, n = 20):
    missing_rows = np.where(np.isnan(sum(X.T)) == True)[0]
    indices = np.random.randint(low=0, high=len(missing_rows), size=(n,))
    ids =  missing_rows[indices]

    X_truth_, X_ = X_truth[ids], X[ids]
    model = {'mu' : imputed_mu, 'Sigma' : imputed_Sigma}

    print("NaN Rate :", nan_rate)
    print("mu :", mu)
    print("Sigma:\n", Sigma)
    print("----------------------------------------------------------------------------------------------------")
    print("Imputed mu :",imputed_mu)
    print("Imputed Sigma:\n",imputed_Sigma)
    print("----------------------------------------------------------------------------------------------------")

    np.set_printoptions(precision=3) 
    print("%-30s\t\t%-30s\t\t%-30s" %("x_truth", "x_missing", "x_imputed"))
    print("----------------------------------------------------------------------------------------------------")
    for i in range(len(ids)): 
        x = X_[i]
        x_truth = X_truth_[i]
        x_tilde, _ = gaussCondition(model, x)
        print("%-30s\t\t%-30s\t\t%-30s" %(str(x_truth), str(x), str(x_tilde.flatten())))
    
    print("----------------------------------------------------------------------------------------------------")

 

def main():
    nan_rates = np.linspace(0.20, 0.50, 10)
    imputed_mu_list = []
    imputed_Sigma_list = []
    observed_mu_list = []
    observed_Sigma_list = []
    
    nrows = 2 
    ncols = 2
    m, n = 0, 4
    idx_list = list(np.random.randint(low=0, high=len(nan_rates), size=(n,)))
    
    figure, axes = plt.subplots(nrows, ncols)
    
    i, j = 0, 0
    for nan_rate in nan_rates:
        result = simulate_nan(nan_rate)
        X = result["X"].copy()
        X_truth = result['X_truth']
        mu = result['mu_truth']
        Sigma = result['Sigma_truth']

        result, logprobs_hist = impute_em(X, nan_rate, result['nan_rate_actual'])
        imputed_mu = result['imputed_mu']
        imputed_Sigma = result['imputed_Sigma']
        fill_values(X_truth, X, mu, Sigma, imputed_mu, imputed_Sigma, nan_rate, 10)

        if m in idx_list:
            plot_observed_likelihood(figure, axes,i, j, logprobs_hist,nan_rate)
            j += 1
            if j >= ncols:
                j = 0
                i += 1
        m += 1

        observed_mu = result['observed_mu']
        observed_Sigma = result['observed_Sigma']

        imputed_mu_list.append(imputed_mu)
        imputed_Sigma_list.append(imputed_Sigma)
        observed_mu_list.append(observed_mu)
        observed_Sigma_list.append(observed_Sigma)

    figure.tight_layout()  
    plt.show()
    plot_imputation(mu, Sigma, imputed_mu_list, imputed_Sigma_list,
                            observed_mu_list, observed_Sigma_list, nan_rates)    


if __name__ == '__main__':
    main()




