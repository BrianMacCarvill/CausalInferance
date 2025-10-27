from typing import TypeVar

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from typing import Protocol

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
    def trace_method(
        self,
    ):
        
        pass