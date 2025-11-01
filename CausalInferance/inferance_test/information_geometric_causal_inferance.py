from typing import TypeVar, Tuple, Dict

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from typing import Protocol

frame = TypeVar('frame', NDArray, pd.Series, pd.DataFrame)

class IGCI:
    
    def __init__(
        self,
        x: frame,
        y: frame,
        ):
        pass
    
    def igci_estimator(
        a,
        b
    ) -> float:
        """
        IGCI core equation
        
        Used terms "a" and "b" instead of "x" and "y" to avoid confusion of say _igci_estimator(x=y, y=x)
        """
        
        order_a = np.argsort(a)
        
        a = a[order_a]
        b = a[order_b]
        
        denumerator = abs(np.diff(a))
        numinator = abs(np.diff(b))
        
        c_a_b = sum(np.log(numinator / denumerator)) / (len(numinator) - 1)  
        
        return c_a_b
    
    @staticmethod
    def main(
        x: frame,
        y: frame,
        unit_scale: bool = True,
        unit_var: bool = False,
        as_dict: bool = False
    ) -> Tuple(float, float) | Dict[str: float]:
        """Calculates the Information Geometric Causal Inference
        
        For more details see: https://jmlr.org/papers/v17/14-518.html page 23 

        Args:
            x (frame): _description_
            y (frame): _description_
            unit_scale (bool, optional): _description_. Defaults to True.
            unit_var (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple(float, float) | Dict[str: float]: _description_
        """        
        
        if len(x) != len(y):
            ValueError("x and y must be the same size")
            
        if unit_scale & unit_var:
            ValueError("unit_scale and unit_var cannot be True at the same time")
        
        #TODO: make this useful for pandas stuff
        # isinstance(ex, np.ndarray)
        if unit_scale:
            x = _unit_range(x)
            y = _unit_range(y)
            
        if unit_var:
            x /= np.std(x)
            y /= np.std(y)
            
        order_x = np.argsort(x)
        order_y = np.argsort(y)
        
        c_x_y, c_y_x = igci_estimator(a=x[order_x], b=y[order_x]), igci_estimator(a=y[order_y], b=x[order_y])
        
        if as_dict:
            return {"c_x_y": c_x_y, "c_y_x": c_y_x}
        else:
            return c_x_y, c_y_x