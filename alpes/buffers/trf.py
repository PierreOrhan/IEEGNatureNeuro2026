import numpy as np
import typing as tp
from pydantic import BaseModel, Field
import warnings
from typing import List
from .api import AbstractBuffer

def generate_tapAdd_features(features:np.ndarray, t_current:tp.List[float], period:float, nb_buffer:int,
                           prediction_start:float, prediction_end:float, frequency:int,duration:float)->tp.List[np.ndarray]:
    """
        TRF features
        prediction_start: phase
        prediction_end: phase + duration = phase + period*nb_buffer
        period: duration of the period of each subspace
        nb_buffer: number of subspace in the buffer
        duration: total duration of the recording, in s
    """
    
    tap_features= [np.zeros((int(np.ceil(duration*frequency)), features.shape[1]))+np.nan for _ in range(nb_buffer)]
    ## for each word in the dataset, we write onto the tap directly (no real buffer in this case):
    word_id = 0
    for time_w in t_current: # for each word in the sentence:
        start_wbuffer = int((time_w + prediction_start) * frequency)
        # end_wbuffer = int((time_w + prediction_end) * frequency)
        tap_len = int(np.ceil(period * frequency))
        for tap_id in range(nb_buffer):
            b = tap_features[tap_id][start_wbuffer+tap_id*tap_len:start_wbuffer+(tap_id+1)*tap_len,:]
            tap_features[tap_id][start_wbuffer+tap_id*tap_len:start_wbuffer+(tap_id+1)*tap_len,:][np.isnan(b)] = 0
            tap_features[tap_id][start_wbuffer+tap_id*tap_len:start_wbuffer+(tap_id+1)*tap_len,:] += features[word_id]
        word_id+=1
    print("End generating tap features...")

    print("Removing all nan feature")
    good_feature = []
    for tap_id in range(nb_buffer):
        if not np.all(np.isnan(tap_features[tap_id])):
            good_feature += [tap_id]
    tap_features = [tap_features[tap_id] for tap_id in good_feature]    
    return tap_features

def generate_reverse_tapAdd_features(features:np.ndarray, t_current:tp.List[float], period:float, nb_buffer:int,
                           prediction_start:float, prediction_end:float, frequency:int,duration:float)->tp.List[np.ndarray]:
    """
        TRF features.
        prediction_start: phase
        prediction_end: phase + duration = phase + period*nb_buffer
        period: duration of the period of each subspace
        nb_buffer: number of subspace in the buffer
        duration: total duration of the recording, in s
    """
    tap_features= [np.zeros((int(np.ceil(duration*frequency)), features.shape[1]))+np.nan for _ in range(nb_buffer)]
    ## for each word in the dataset, we write onto the tap directly (no real buffer in this case):
    word_id = 0
    for time_w in t_current: # for each word in the sentence:
        start_wbuffer = int((time_w + prediction_start - period*nb_buffer) * frequency)
        # end_wbuffer = int((time_w + prediction_end - period*nb_buffer) * frequency)
        tap_len = int(np.ceil(period * frequency))
        for tap_id in range(nb_buffer):
            b = tap_features[tap_id][start_wbuffer+tap_id*tap_len:start_wbuffer+(tap_id+1)*tap_len,:]
            tap_features[tap_id][start_wbuffer+tap_id*tap_len:start_wbuffer+(tap_id+1)*tap_len,:][np.isnan(b)] = 0
            tap_features[tap_id][start_wbuffer+tap_id*tap_len:start_wbuffer+(tap_id+1)*tap_len,:] += features[word_id]
        word_id+=1
    print("End generating reverse tap features...")

    print("Removing all nan feature")
    good_feature = []
    for tap_id in range(nb_buffer):
        if not np.all(np.isnan(tap_features[tap_id])):
            good_feature += [tap_id]
    tap_features = [tap_features[tap_id] for tap_id in good_feature]    
    return tap_features

class TapAdd(AbstractBuffer):
    """
        Generate features according to a tap order. 
        Each subspace is supposed to predict the neural activity for a "period" duration.
        Then the next subspace should predict it.
        For each word, the subspace supposed to predict the neural activity are locked relative to the word offset.
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer")
    period: float = Field(..., description="Duration of the period in s; " \
    "this is the time relative to the word during which the subspace is supposed to predict " \
    "the neural activity before switching to the next subspace.")
    def __call__(self,features:np.ndarray, t_current:tp.List[float],context:tp.List[tp.Any])->tp.List[np.ndarray]:
        assert self.period == 1/self.frequency, "The period should be equal to the inverse of the frequency for TRF features."
        print("Generating tap features...")
        # if np.any(np.diff(np.array(t_current)) < self.period):
        #     number_error = np.sum(np.diff(np.array(t_current)) < self.period)
        #     warnings.warn(f"The period is longer than the time between two consecutive words ({number_error} occurrences). " \
        #     "In the case of an error, we overwrite the tap with to the latest word!")
        tap_features = generate_tapAdd_features(features,t_current,self.period,self.nb_buffer,self.prediction_start,self.prediction_end,self.frequency,self.duration)  
        return tap_features


class ReverseTapAdd(AbstractBuffer):
    """
        Generate features according to a reverse TRF order. 
        Each subspace is supposed to predict the neural activity for a "period" duration.
        Then the next subspace should predict it.
        For each word, the subspace supposed to predict the neural activity are locked relative to the word offset.
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer")
    period: float = Field(..., description="Duration of the period in s; " \
    "this is the time relative to the word during which the subspace is supposed to predict " \
    "the neural activity before switching to the next subspace.")
    def __call__(self,features:np.ndarray, t_current:tp.List[float],
                 context:tp.List[tp.Any])->tp.List[np.ndarray]:
        # print("Generating tap features...")
        # if np.any(np.diff(np.array(t_current)) < self.period):
        #     number_error = np.sum(np.diff(np.array(t_current)) < self.period)
        #     warnings.warn(f"The period is longer than the time between two consecutive words ({number_error} occurrences). " \
        #     "In the case of an error, we overwrite the tap with to the latest word!")
        tap_features = generate_reverse_tapAdd_features(features,t_current,self.period,self.nb_buffer,self.prediction_start,self.prediction_end,self.frequency,self.duration)  
        return tap_features

class SymTapAdd(AbstractBuffer):
    """
        Reverse Tap to Tap features.
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer")
    period: float = Field(..., description="Duration of the period in s; " \
    "this is the time relative to the word during which the subspace is supposed to predict " \
    "the neural activity before switching to the next subspace.")
    def __call__(self,features:np.ndarray, t_current:tp.List[float],context:tp.List[tp.Any])->tp.List[np.ndarray]:
        # print("Generating tap features...")
        # if np.any(np.diff(np.array(t_current)) < self.period):
        #     number_error = np.sum(np.diff(np.array(t_current)) < self.period)
        #     warnings.warn(f"The period is longer than the time between two consecutive words ({number_error} occurrences). " \
        #     "In the case of an error, we overwrite the tap with to the latest word!")
        past_tap_features = generate_reverse_tapAdd_features(features,t_current,self.period,self.nb_buffer,self.prediction_start,self.prediction_end,self.frequency,self.duration)  
        future_tap_features = generate_tapAdd_features(features,t_current,self.period,self.nb_buffer,self.prediction_start,self.prediction_end,self.frequency,self.duration)
        return past_tap_features+future_tap_features


class SymTapAdd_sharedreg(AbstractBuffer):
    """
        Reverse Tap to Tap features.
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer")
    period: float = Field(..., description="Duration of the period in s; " \
    "this is the time relative to the word during which the subspace is supposed to predict " \
    "the neural activity before switching to the next subspace.")
    def __call__(self,features:np.ndarray, t_current:tp.List[float],context:tp.List[tp.Any])->tp.List[np.ndarray]:
        # print("Generating tap features...")
        # if np.any(np.diff(np.array(t_current)) < self.period):
        #     number_error = np.sum(np.diff(np.array(t_current)) < self.period)
        #     warnings.warn(f"The period is longer than the time between two consecutive words ({number_error} occurrences). " \
        #     "In the case of an error, we overwrite the tap with to the latest word!")
        past_tap_features = generate_reverse_tapAdd_features(features,t_current,self.period,self.nb_buffer,self.prediction_start,self.prediction_end,self.frequency,self.duration)  
        future_tap_features = generate_tapAdd_features(features,t_current,self.period,self.nb_buffer,self.prediction_start,self.prediction_end,self.frequency,self.duration)
        
        out_features = np.concatenate(past_tap_features + future_tap_features, axis=1)
        return [out_features]