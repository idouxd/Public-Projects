import pandas as pd
import numpy as np
from scipy.optimize import dual_annealing

def assign_factors(df, factors):
    '''
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
    Parameters:
        df (pd.DataFrame): dataframe containing car_age_freq_factor & drv_age_freq_factor

    Returns:
        gini: gini under a given set of factors
    '''
    #Offbalance calc freq to actual freq
    avg_freq = df['ClaimNb'].sum()/df['Exposure'].sum()
    adjusted_df = df.copy()
    adjusted_df['car_and_drv_freq'] = adjusted_df['car_age_freq_factor']*adjusted_df['drv_age_freq_factor']
    obs_freq = (adjusted_df['car_and_drv_freq']*adjusted_df['Exposure']).sum()/adjusted_df['Exposure'].sum()
    offbalance = avg_freq/obs_freq
    adjusted_df['rebalanced_car_and_drv_freq'] = offbalance*adjusted_df['car_and_drv_freq']

    #Gini calculation
    sorted_df = adjusted_df.sort_values(by='rebalanced_car_and_drv_freq', ascending=True).reset_index(drop=True)
    sorted_values = sorted_df['rebalanced_car_and_drv_freq'].values
    sorted_weights = sorted_df['Exposure'].values

    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    weighted_values = sorted_values * sorted_weights
    cum_weighted_values = np.cumsum(weighted_values)
    total_weighted_value = cum_weighted_values[-1]
    
    # Gini calculation
    weighted_gini_coeff = (
        (cum_weights / total_weight) * cum_weighted_values
    ).sum() / (total_weight * total_weighted_value)
    
    return 1 - 2 * weighted_gini_coeff

def solve_factors(df, lower_limit, upper_limit):
    '''
    Parameters:
        df (pd.DataFrame): Input DataFrame

    '''
    def objective_function(factors):
        adjusted_df = assign_factors(df, factors)
        return calculate_loss(adjusted_df)
    
    bounds = list(zip(lower_limit, upper_limit))
    result = dual_annealing(objective_function, bounds, maxfun=999)

    return result

full_df = pd.read_csv('freMTPL2freq.csv')
df = full_df.sample(frac=.8).reset_index(drop=True)
df['freq'] = df['ClaimNb'] / df['Exposure']
lower_limit = (0.5,0.5,0.5,0.5,0.5,0.5)
upper_limit = (3,3,3,3,3,3)
answer = solve_factors(df, lower_limit, upper_limit)
print(answer)
