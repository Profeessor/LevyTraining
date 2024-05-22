import pandas as pd
import scipy.linalg as sp
import numpy as np
def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

from scipy.stats import pearsonr

from scipy.stats import chi2
chi2.ppf((1-0.001), df=2)
#> 9.21

def calculate_corr(df):
    dfcols = pd.DataFrame(columns=df.columns)
    corr = dfcols.transpose().join(dfcols, how='outer')
    corr_frame = pd.DataFrame()
    ind_all = []
    for r in df.columns:
        corr_frame = pd.DataFrame()

        if r == 'Experiment':
        
            continue

        for c in df.columns:
            if c == r or c == 'Experiment':
                continue
            
            
        
            data = {'Experiment':df['Experiment'],r:np.array(df[r]),c:np.array(df[c])}
           # print(data)
            corr_frame = pd.DataFrame(data)

            df_x = corr_frame[[r, c]]
            df_x['mahala'] = mahalanobis(x=df_x, data=corr_frame[[r, c]])
            df_x['p_value'] = 1 - chi2.cdf(df_x['mahala'], 2)

        # Extreme values with a significance level of 0.01
            df_outliers=df_x.loc[df_x.p_value < 0.001]
            ind=df_outliers.index
            ind_all.append(ind.shape)
        #drop the outliers with index     
            corr_without_outlier = corr_frame.drop(ind)
            x = corr_without_outlier[r]
            y = corr_without_outlier[c]

            


            
            
            corr[r][c] = round(pearsonr(x, y)[0], 4)
    print(ind_all)
    return corr       
            
