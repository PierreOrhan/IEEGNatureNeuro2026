from pydantic import BaseModel, Field
import warnings
from typing import List
import numpy as np
import typing as tp
from .api import AbstractBuffer


def generate_reverse_bufferAdd_features(features:np.ndarray, t_current:tp.List[float], 
                             duration:float,frequency:float,nb_buffer:int,
                             prediction_start:np.ndarray[float],prediction_end:np.ndarray[float])->tp.List[np.ndarray]:
    """
        Method for generating buffer features according to a buffer order.
    """
    buffer_features= [np.zeros((int(np.ceil(duration*frequency)), features.shape[1]))+np.nan for _ in range(nb_buffer)]
        
    t_current = np.array(t_current)
    assert len(features)==len(t_current), "The number of features should be equal to the total number of elements in the dataset."
    ## for each word in the dataset, we write onto the buffer:
    word_id = 0
    all_pcurrent = t_current + prediction_start
    all_pend = t_current + prediction_end
    simul_buffer =  [[] for _ in range(buffer_features[0].shape[0])]
    for pstart, pend in zip(all_pcurrent, all_pend):
        start_wbuffer = int(pstart * frequency)
        end_wbuffer = int(pend * frequency)
        for s in range(max(0,min(start_wbuffer, buffer_features[0].shape[0]-1)), min(end_wbuffer, buffer_features[0].shape[0])):
            simul_buffer[s] += [features[word_id]]
        word_id += 1

    ## Unpops the simul_buffer to generate the buffer features
    for t in range(buffer_features[0].shape[0]):
        # if len(simul_buffer[t])>nb_buffer:
        #     warnings.warn(f"The number of elements in the buffer at time {t/frequency}s is greater than the number of subspaces. This may lead to unexpected results. In this case the activity are added into the last buffer.")
        for b in range(min(len(simul_buffer[t]),nb_buffer)):
            buffer_features[b][t,:] = simul_buffer[t][b]
        ## Add mechanism for writing error:
        if len(simul_buffer[t])>nb_buffer:
            for b in range(nb_buffer,len(simul_buffer[t])):
                buffer_features[-1][t,:] += simul_buffer[t][b]
    return buffer_features

def generate_bufferAdd_features(features:np.ndarray, t_current:tp.List[float],
                                    duration:float,frequency:float,nb_buffer:int,
                                    prediction_start:np.ndarray[float],prediction_end:np.ndarray[float])->tp.List[np.ndarray]:
    """
        Method for generating buffer features according to a buffer order.
    """
    buffer_features= [np.zeros((int(np.ceil(duration*frequency)), features.shape[1]))+np.nan for _ in range(nb_buffer)]
        
    ## for each word in the dataset, we write onto the buffer:
    word_id = 0 
    t_current = np.array(t_current)
    assert len(features)==len(t_current), "The number of features should be equal to the total number of elements in the dataset."

    all_pcurrent = t_current + prediction_start
    all_pend = t_current + prediction_end
    simul_buffer =  [[] for _ in range(buffer_features[0].shape[0])]
    for pstart, pend in zip(all_pcurrent, all_pend):
        start_wbuffer = int(pstart * frequency)
        end_wbuffer = int(pend * frequency)
        for s in range(max(0,min(start_wbuffer, buffer_features[0].shape[0]-1)), min(end_wbuffer, buffer_features[0].shape[0])):
            simul_buffer[s] = [features[word_id]] + simul_buffer[s]
        word_id += 1

    ## Unpops the simul_buffer to generate the buffer features
    for t in range(buffer_features[0].shape[0]):
        # if len(simul_buffer[t])>nb_buffer:
        #     warnings.warn(f"The number of elements in the buffer at time {t/frequency}s is greater than the number of subspaces. This may lead to unexpected results. In this case we consider the buffer as full and the word is not processed at this time point.")
        for b in range(min(len(simul_buffer[t]),nb_buffer)):
            buffer_features[b][t,:] = simul_buffer[t][b]

        ## Add mechanism for writing error:
        if len(simul_buffer[t])>nb_buffer:
            for b in range(nb_buffer,len(simul_buffer[t])):
                buffer_features[-1][t,:] += simul_buffer[t][b]
    print("End generating reverse buffer features...")
    return buffer_features


class BufferAdd(AbstractBuffer):
    """
        Generate features according to a buffer order. 
        Elements are put at the end of the buffer, and if occupied at a previous position.
        At the end of the element processing, they are moved out of the buffer. 
        All elements in the buffer are then moved into the preceding subspace.
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer")
    def __call__(self,features:np.ndarray, t_current:List[float])->tp.List[np.ndarray]:
        print("Generating buffer features...")
        ## pre-allocate:
        buffer_features = generate_bufferAdd_features(features,t_current,self.duration,self.frequency,self.nb_buffer,self.prediction_start,self.prediction_end)
        print("End generating buffer features...")
        return buffer_features
    
class ReverseBufferAdd(AbstractBuffer):
    """
        Generate features according to a reverse-buffer order. 
        Elements are put at the start of the buffer, and if occupied all elements are moved to the right of the buffer.
        At the end of the element processing, they are moved out of the buffer.
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer")
    def __call__(self,features:np.ndarray, t_current:tp.List[float],context:tp.List[tp.Any])->tp.List[np.ndarray]:
        print("Generating reverse buffer features...")
        ## pre-allocate:
        buffer_features = generate_reverse_bufferAdd_features(features,t_current,self.duration,self.frequency,self.nb_buffer,self.prediction_start,self.prediction_end)
        return buffer_features

class SymBufferAdd(AbstractBuffer):
    """
        Buffer --> Reverse buffer pile
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer, both for the positive and negative time")
    buffer_duration: float = Field(..., description="Duration of the past buffer (Buffer) in s")
    def __call__(self,features:np.ndarray, t_current:tp.List[float],context:tp.List[tp.Any])->tp.List[np.ndarray]:
        print("Generating double buffer features...")
        ## pre-allocate:
        future_buffer_features = generate_bufferAdd_features(features,t_current,self.duration,self.frequency,
                                                   self.nb_buffer,self.prediction_start,self.prediction_end)
        past_buffer_features = generate_reverse_bufferAdd_features(features,t_current,self.duration,self.frequency,self.nb_buffer,
                                                                self.prediction_start-self.buffer_duration,
                                                                self.prediction_end-self.buffer_duration)
        print("End generating double buffer features...")
        return past_buffer_features + future_buffer_features

class SymBufferAdd_sharedreg(SymBufferAdd):
    """
        Buffer --> Reverse buffer pile
    """
    def __call__(self,features:np.ndarray, t_current:tp.List[float],context:tp.List[tp.Any])->tp.List[np.ndarray]:
        print("Generating double buffer features...")
        out = super().__call__(features, t_current,context)
        return [np.concatenate(out, axis=1)]
