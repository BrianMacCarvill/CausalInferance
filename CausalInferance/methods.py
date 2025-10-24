from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from typing import TypeVar

frame = TypeVar('frame', NDArray, pd.Series, pd.DataFrame)

from dataclasses import dataclass

@dataclass
class CausalDataClass:
    ...

class Causality:
    """
    Class to define causality
    """
        
    def fit_linear_reg(X: frame, y: frame, **kwargs) -> None:
        
        self.model = LinearRegression(**kwargs)
        
        self.model.fit(X,y)
        
    def fit_random_forest_reg(X: frame, y: frame, **kwargs) -> None:
        
        self.model = RandomForestRegressor(**kwargs)
        
        self.model.fit(X,y)
        
    def fit_custom_reg(X: frame, y: frame, model: object, **kwargs) -> None:
        
        self.model = model(**kwargs)
        
        self.model.fit(X,y)
        
    def is_causal(X, y) -> float:
        
        residuals = y - self.model.predict(X)
        
        p_val = pearsonr(residuals, x)[1]
        
        return p_val

    def is_causal_two_sided(X, y, model = 'linear', return_residuals: bool = False):
        """
        
        """
        
        if model == 'linear':
            model = LinearRegression()
            
        model.fit(X,y)
        
        print((model.coef_, model.intercept_))
        
        residuals_X = y - model.predict(X)
        
        p_val_X = pearsonr(residuals_X, X)
        
        model.fit(y,X)
        
        residuals_y = X - model.predict(y)
        
        p_val_y = pearsonr(residuals_y, y)
        
        print((model.coef_, model.intercept_))
        
        if return_residuals:
            return p_val_X, residuals_X, p_val_y, residuals_y
        return p_val_X, p_val_y
    
if __name__ == '__main__':
    
    from numpy.random import normal
    
    import matplotlib.pyplot as plt
    
    n = 10_000
    
    x = normal(size = 10_000)
    
    y = 2*x + normal(size = 10_000)
    
    x = x.reshape(-1, 1)
    
    y = y.reshape(-1, 1)
    
    a = Causality.is_causal_two_sided(x,y)
    
    print(a)