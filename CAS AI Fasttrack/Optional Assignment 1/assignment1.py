import pandas as pd
import numpy as np
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping
from skopt import gp_minimize
from skopt.space import Real


def assign_factors(df, factors):
    '''
    For a given set of factors, map those factors according to the rules below. 

    Parameters:
        df (pd.DataFrame): input dataframe
        factors (tuple): a tuple of 6 numeric factors (a,b,c,d,e,f)

    Returns:
        pd.DataFrame containing car_age_freq_factor & drv_age_freq_factor
    '''
    if not isinstance(factors, tuple) or len(factors) != 6:
        raise ValueError('Factors must be a tuple of six numeric values!')
    
    adjusted_df = df.copy()
    a,b,c,d,e,f = factors

    #Car Age Frequency
    conditions = [(adjusted_df['VehAge']>=0)&(adjusted_df['VehAge']<=7),
                  (adjusted_df['VehAge']>=8)&(adjusted_df['VehAge']<=17),
                  (adjusted_df['VehAge']>=18)]
    values = [a,b,c]
    adjusted_df['car_age_freq_factor'] = np.select(conditions,values,default=np.nan)

    #Driver Age Frequency
    conditions = [(adjusted_df['DrivAge']>=18)&(adjusted_df['DrivAge']<=25),
                  (adjusted_df['DrivAge']>=26)&(adjusted_df['DrivAge']<=50),
                  (adjusted_df['DrivAge']>=50)]
    values = [d,e,f]
    adjusted_df['drv_age_freq_factor'] = np.select(conditions,values,default=np.nan)

    return adjusted_df

def calculate_loss(df):
    '''
    For a given set of factors (a,b,c,d,e,f) we calculate a weighted gini

    Parameters:
        df (pd.DataFrame): dataframe containing car_age_freq_factor & drv_age_freq_factor

    Returns:
        gini: gini under a given set of factors
    '''
    #Offbalance calc freq to actual freq
    avg_freq = df['ClaimNb'].sum()/df['Exposure'].sum()
    adjusted_df = df.copy()[['Exposure','freq','car_age_freq_factor','drv_age_freq_factor']]
    adjusted_df['car_and_drv_freq'] = adjusted_df['car_age_freq_factor']*adjusted_df['drv_age_freq_factor']
    obs_freq = (adjusted_df['car_and_drv_freq']*adjusted_df['Exposure']).sum()/adjusted_df['Exposure'].sum()
    offbalance = avg_freq/obs_freq
    adjusted_df['rebalanced_car_and_drv_freq'] = offbalance*adjusted_df['car_and_drv_freq']

    #Gini calculation
    x = np.asarray(adjusted_df['rebalanced_car_and_drv_freq'])
    w = np.asarray(adjusted_df['Exposure'])
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_w = w[sorted_indices]
    # Force float dtype to avoid overflows
    cumw = np.cumsum(sorted_w, dtype=float)
    cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
    gini = (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1]))

    return -1*gini

def solve_factors(df, lower_limit, upper_limit, method='da'):
    '''
    Optimizes factors (a, b, c, d, e, f) for the assign_factors function to minimize loss.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        lower_limit (tuple): tuple of lower bound of factors
        upper_limit (tuple): tuple of higher bound of factors
        method (string): chosen method (dual annealing = 'da', basin hopping = 'bh', or bayesian optimization = 'bo')

    Returns:
        OptimizeResult: Result of the optimization process.
    '''
    def objective_function(factors):
        factors_tuple = tuple(factors)
        adjusted_df = assign_factors(df, factors_tuple)
        return calculate_loss(adjusted_df)
    
    bounds = list(zip(lower_limit, upper_limit))
    if method == 'da':
        result = dual_annealing(objective_function, bounds, maxfun=999)
    elif method == 'bh':
        result = basinhopping(objective_function,x0=np.random.uniform(lower_limit, upper_limit), niter=999, T=1.0, stepsize=0.1)
    elif method == 'bo':
        dimensions = [Real(low, high) for low, high in zip(lower_limit, upper_limit)]
        result = gp_minimize(objective_function, dimensions, n_calls=999)
    else:
        raise ValueError('Wrong Method')

    return result

full_df = pd.read_csv('freMTPL2freq.csv')

df = full_df.sample(frac=.8).reset_index(drop=True)
df['freq'] = df['ClaimNb'] / df['Exposure']
lower_limit = (0.5,0.5,0.5,0.5,0.5,0.5)
upper_limit = (3.0,3.0,3.0,3.0,3.0,3.0)
# answer = solve_factors(df, lower_limit, upper_limit, 'da')

# print("Dual Annealing Optimal Factors:", [round(value, 4) for value in answer.x])
# print("Dual Annealing Maximum Gini:", -1*round(answer.fun,4))

# answer = solve_factors(df, lower_limit, upper_limit, 'bh')

# print("Basin Hopping Optimal Factors:", [round(value, 4) for value in answer.x])
# print("Basin Hopping Maximum Gini:", -1*round(answer.fun,4))

answer = solve_factors(df, lower_limit, upper_limit, 'bo')

print("Bayesian Optimization Optimal Factors:", [round(value, 4) for value in answer.x])
print("Bayesian Optimization Maximum Gini:", -1*round(answer.fun,4))

