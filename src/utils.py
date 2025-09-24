import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.impute import SimpleImputer

def fix_headers(cols):
    """lower and replace special chars in column headers"""
    replace_chars = str.maketrans({'-': '', ' ': '_', '/': '', '(': '', ')': ''})
    return [col.lower().translate(replace_chars).replace('__', '_') for col in cols]

def fix_dtypes(df, cols):
    """convert columns to numeric dtype"""
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df

def remove_negative(df, cols):
    """remove any row with negative values"""
    mask = (df[cols] < 0).any(axis=1)
    return df.loc[~mask]

def get_desired_color_interval(color_series: pd.Series):
    """calculates the median confidence interval for the color variable"""
    median = color_series.median()
    MAD = stats.median_abs_deviation(color_series)
    desired_interval = median - 2 * MAD, median + 2 * MAD
    return desired_interval

def feature_engineering(df):
    """create new features based on existing ones"""
    df_copy = df.copy()
    df_copy['base_malt_amount_total'] = df_copy['1st_malt_amount_kg'] + df_copy['2nd_malt_amount_kg']
    df_copy['malt_amount_total'] = df_copy['base_malt_amount_total'] + df_copy['roast_amount_kg']
    df_copy['1st_malt_amount_ratio'] = df_copy['1st_malt_amount_kg'] / df_copy['base_malt_amount_total']
    df_copy['roast_malt_amount_ratio'] = df_copy['roast_amount_kg'] / df_copy['base_malt_amount_total']
    df_copy['weighted_malt_color_amount_total'] = (df_copy['1st_malt_amount_kg'] * df_copy['1st_malt_color']) + (df_copy['2nd_malt_amount_kg'] * df_copy['2nd_malt_color']) + (df_copy['roast_amount_kg'] * df_copy['roast_color'])
    df_copy['weighted_malt_color_amount_roast'] = (df_copy['roast_amount_kg'] * df_copy['roast_color']) / df_copy['weighted_malt_color_amount_total']
    df_copy['weighted_malt_color_amount_1st'] = (df_copy['1st_malt_amount_kg'] * df_copy['1st_malt_color']) / df_copy['weighted_malt_color_amount_total']
    df_copy['mt_cumulative_heat'] = df_copy['mt_temperature'] * df_copy['mt_time']
    df_copy['wk_cumulative_heat'] = df_copy['wk_temperature'] * df_copy['wk_time']
    df_copy['total_cumulative_heat'] = df_copy['mt_cumulative_heat'] + df_copy['wk_cumulative_heat']
    df_copy['mt_time_per_amount'] = df_copy['mt_time'] / df_copy['malt_amount_total']
    df_copy['wk_time_per_amount'] = df_copy['wk_time'] / df_copy['malt_amount_total']
    df_copy['mt_temperature_per_amount'] = df_copy['mt_temperature'] / df_copy['malt_amount_total']
    df_copy['wk_temperature_per_amount'] = df_copy['wk_temperature'] / df_copy['malt_amount_total']
    df_copy['whp_transfer_ratio'] = df_copy['whp_transfer_time'] / df_copy['whp_rest_time']
    df_copy['day_of_week'] = df_copy['datetime'].dt.day_of_week
    df_copy['is_weekend'] = np.where(df_copy['day_of_week'].isin([5, 6]), 1, 0)
    return df_copy

class PreprocessData:
    """wrapper of pre-processing functions to be used with a sklearn-style pipeline
    
    1 step - impute missing features with the median of the test set
    2 step - call feature_engineering to make new features
    """
    def __init__(self, cols: list):
        self.cols = cols
        self.imputer = SimpleImputer(strategy='median')
    
    def fit(self, X: pd.DataFrame, y=None):
        """fit imputer only on training data"""
        self.imputer.fit(X[self.cols])
        return self
    
    def transform(self, X: pd.DataFrame):
        """returns concatenated df"""
        X_imputed = self.imputer.transform(X[self.cols])
        headers = self.imputer.get_feature_names_out()
        df_imputed = pd.DataFrame(X_imputed, columns=headers)
        x_reset = X.reset_index(drop=True)
        new_df =  pd.concat([x_reset.drop(columns=self.cols), df_imputed], axis=1)
        new_df = feature_engineering(new_df)
        new_df = new_df.drop(columns=['datetime'])
        return new_df