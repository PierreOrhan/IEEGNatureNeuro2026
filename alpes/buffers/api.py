import numpy as np
import typing as tp
from pydantic import BaseModel, Field
import warnings
from typing import List
import tqdm

class AbstractBuffer(BaseModel):
    """
    Abstract class for generating features according to a tap, buffer, or reverse buffer order. 
    The class should be initialized with the desired parameters (e.g., buffer size, tap size, etc.) and then called with the features to generate the new features according to the specified order.
    """
    prediction_start: float = Field(..., description="Start of the prediction in s; this is the time relative to the word at which the subspace is supposed to start predicting the neural activity")    
    prediction_end: float = Field(..., description="End of the prediction in s; this is the time relative to the word at which the subspace is supposed to stop predicting the neural activity")
    frequency: int = Field(..., description="Frequency of the neural data in Hz; this is used to convert the prediction start and end times from s to samples.")
    duration : float = Field(..., description="Total duration of the dataset in s; this is used to determine the shape of the output features.")


    def __call__(self,features:np.ndarray, t_current:tp.List[np.ndarray],
                 context:tp.List[tp.Any])->tp.List[np.ndarray]:
        """
        Generate new features according to the specified order.
        Args:
            features: np.ndarray of shape (n_elements, n_features) containing the original features to be transformed.
            t_current: List of float, containing the start/offset time (s) of the elements.
            context: List, containing the context information for each element.
        Returns:
            List[np.ndarray], each of shape (duration*frequency,n_new_features) containing the new features generated according to the specified order.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
