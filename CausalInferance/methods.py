from typing import TypeVar, Callable, Tuple
from dataclasses import dataclass
import importlib

import numpy as np
from numpy.typing import NDArray
import pandas as pd

frame = TypeVar('frame', NDArray, pd.Series, pd.DataFrame)

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from CausalInferance.xicor import xicor

from CausalInferance.registry_utils import _get_from_registry

REGISTRY_MODELS: dict[str, Callable] = {
    "linear": LinearRegression,
    "random_forest": RandomForestRegressor,
}

REGISTRY_CORR_METRICS: dict[str, Callable] = {
    "pearson": pearsonr,
    "spearman": spearmanr,
    "xicor": xicor
}

@dataclass
class CausalDataClass:
    ...

class Causality:
    """
    Class to define causality between 2 variables
    
    Within it there are methods 
    """

    def is_causal_two_sided(
        X, 
        y, 
        *, 
        reg_model: str | Callable = 'linear', 
        corr_metric: str | Callable = 'pearson', 
        return_residuals: bool = False, 
        **kwargs 
        ) -> Tuple: 
        
        """
        method for inferring causality based on an additive noise model
        
        Args:
            X (frame): The first list of realised variables
            y (frame): The second list of realised variables
            reg_model (str | Callable, optional): A string specifying the regression model used or a Callable with the methods `fit` and `predict` sklearn style
            corr_metric (str | Callable, optional):  A string specifying the correlation metric used or a correlation function to use
            return_residuals (bool): bool to return the regression residuals
            
        Returns:
            bool: The return value. True for success, False otherwise.
            
        Raises:
            ValueError: If `X` and `y` have incompatible shapes or insufficient data points.
            
        Example:
            >>> import numpy as np
            >>> X = np.random.normal(size=100)
            >>> y = 3 * X + np.random.normal(size=100)
            >>> is_causal_two_sided(X, y, model='linear')
        """

        model = _get_from_registry(reg_model, REGISTRY_MODELS, kind="model", **kwargs)

        corr_func = _get_from_registry(corr_metric, REGISTRY_CORR_METRICS, kind="correlation")
        
        #======================================================================
        # fitting X on y
        model.fit(X,y)
                
        residuals_X = y - model.predict(X)
        
        p_val_X = corr_func(residuals_X, X)
        
        #======================================================================
        # fitting y on X
        
        model.fit(y,X)
        
        residuals_y = X - model.predict(y)
        
        p_val_y = corr_func(residuals_y, y)
                
        if return_residuals:
            return p_val_X, residuals_X, p_val_y, residuals_y
        else:
            return p_val_X, p_val_y