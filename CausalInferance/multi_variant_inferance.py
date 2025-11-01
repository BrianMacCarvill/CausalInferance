from typing import TypeVar

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from typing import Protocol

from sklearn.linear_model import LinearRegression

frame = TypeVar('frame', NDArray, pd.Series, pd.DataFrame)

class MultiVariantInferance:
    """
    For multi-dimensional 
    """
    def __init__(
        self,
        x: frame,
        y: frame,
    ):
        self.x = x
        self.y = y

    #TODO: make this a hybrid method
    @staticmethod
    def trace_method(
        x,
        y
    ):
        model = LinearRegression()
        model.fit(x, y)

        cov_x = np.cov(x, rowvar=False)
        cov_y = np.cov(y, rowvar=False)

        numerator = (model.coef_ @ cov_x @ model.coef_.T).trace()
        denumerator = (model.coef_ @ model.coef_.T).trace() * cov_x.trace()
        r_x_y = numerator / denumerator

        model = LinearRegression()
        model.fit(y, x)

        numerator = (model.coef_ @ cov_y @ model.coef_.T).trace()
        denumerator = (model.coef_ @ model.coef_.T).trace() * cov_y.trace()
        r_y_x = numerator / denumerator
        
        return r_x_y, r_y_x