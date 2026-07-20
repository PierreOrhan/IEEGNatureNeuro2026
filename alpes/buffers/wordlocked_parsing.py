from pydantic import BaseModel, Field
import warnings
from typing import List
import numpy as np
import typing as tp
from .api import AbstractBuffer

def generate_bufferAddParseReset_features(features:np.ndarray, 
                                             t_current:tp.List[float],
                                        duration:float,frequency:float,nb_buffer:int,
                                        prediction_start:np.ndarray[float],prediction_end:np.ndarray[float],
                                        context:tp.List[tp.Any])->tp.List[np.ndarray]:
    """
        Method for generating buffer features according to a buffer order, including a parsing mechanism. 
        
        features: (nb_elements, dim, nb_possible_context) array of features for each element in the dataset. The last column allows to modulate the representations depending on the context of elements in the buffer.
        t_current: list of time of each element in the dataset.
        duration: duration of the dataset.
        frequency: frequency of the dataset.
        prediction_start: for each element, the time at which the element starts to be processed relative to t_current.
        prediction_end: for each element, the time at which the element stops to be processed relative to t_current.
        
        context: For each element, adds a contextual information on top of the time.
        context_to_index: given all context of elements in the buffer, returns an index into the last column of the features array.

    """
    buffer_features= [np.zeros((int(np.ceil(duration*frequency)), features.shape[1]))+np.nan for _ in range(nb_buffer)]
        
    ## for each word in the dataset, we write onto the buffer:
    word_id = 0 
    t_current = np.array(t_current)
    assert len(features)==len(t_current), "The number of features should be equal to the total number of elements in the dataset."

    all_pcurrent = t_current + prediction_start
    all_pend = t_current + prediction_end
    simul_buffer =  [[] for _ in range(buffer_features[0].shape[0])]
    context_buffer = [[] for _ in range(buffer_features[0].shape[0])]
    for pstart, pend in zip(all_pcurrent, all_pend):
        start_wbuffer = int(pstart * frequency)
        end_wbuffer = int(pend * frequency)
        
        for s in range(max(0,min(start_wbuffer, buffer_features[0].shape[0]-1)), 
                       min(end_wbuffer, buffer_features[0].shape[0])):
            simul_buffer[s] = [features[word_id]] + simul_buffer[s]
            context_buffer[s] = [word_id] + context_buffer[s]

            ## Reset mechanism:
            if len(context_buffer[s]) > 1 and context[word_id][0] != context[context_buffer[s][1]][0]:  # if the word is the first word of a sentence, we reset the buffer
                simul_buffer[s] = simul_buffer[s][:1]  # resets and keep only the last word in the buffer
                context_buffer[s] = context_buffer[s][:1]  # resets and keep only the last word in the buffer

        word_id += 1
    
    arr_sid = np.array([c[0] for c in context])   # shape (nb_words,)
    arr_wid = np.array([c[1] for c in context])       # shape (nb_words,)

    # generate a dataframe from the context list:
    # dfcontext = pd.DataFrame({"sentence_id":[c[0] for c in context],
    #                           "word_id":[c[1] for c in context]})

    ## Unpops the simul_buffer to generate the buffer features
    for t in range(buffer_features[0].shape[0]):
        # if len(simul_buffer[t])>nb_buffer:
        #     warnings.warn(f"The number of elements in the buffer at time {t/frequency}s is greater than the number of subspaces. This may lead to unexpected results. In this case we consider the buffer as full and the word is not processed at this time point.")
        if len(simul_buffer[t])>0:
            # indexes = context_to_index(context_buffer[t],dfcontext)
                    
            buf_idx = np.array(context_buffer[t])   # indices of words in buffer
            buf_sids = arr_sid[buf_idx]
            buf_wids = arr_wid[buf_idx]

            unique_sids, inv = np.unique(buf_sids, return_inverse=True)
            max_wids = np.full(len(unique_sids), -1, dtype=int)
            np.maximum.at(max_wids, inv, buf_wids)
            indexes = max_wids[inv].tolist()
            # subdf = dfcontext.iloc[context_buffer[t]]
            # subdfg = subdf.groupby("sentence_id").agg("max")
            # indexes = [int(subdfg.loc[k,"word_id"]) for k in subdf["sentence_id"].values]  
            
            for b in range(min(len(simul_buffer[t]),nb_buffer)):
                buffer_features[b][t,:] = simul_buffer[t][b][...,indexes[b]]
        ## Add mechanism for writing error:
        if len(simul_buffer[t])>nb_buffer:
            for b in range(nb_buffer,len(simul_buffer[t])):
                buffer_features[-1][t,:] += simul_buffer[t][b][...,indexes[b]]
    print("End generating reverse buffer features...")
    return buffer_features

def generate_reverse_bufferAddParseReset_features(parser_features:np.ndarray, t_current:tp.List[float], 
                             duration:float,frequency:float,nb_buffer:int,
                             prediction_start:np.ndarray[float],prediction_end:np.ndarray[float],
                             context:tp.List[tp.Any])->tp.List[np.ndarray]:
    """
        Method for generating buffer features according to a buffer order.
    """
    buffer_features= [np.zeros((int(np.ceil(duration*frequency)), parser_features.shape[1]))+np.nan for _ in range(nb_buffer)]
        
    t_current = np.array(t_current)
    assert len(parser_features)==len(t_current), "The number of features should be equal to the total number of elements in the dataset."
    ## for each word in the dataset, we write onto the buffer:
    word_id = 0
    all_pcurrent = t_current + prediction_start
    all_pend = t_current + prediction_end
    simul_buffer =  [[] for _ in range(buffer_features[0].shape[0])]
    context_buffer = [[] for _ in range(buffer_features[0].shape[0])]
    for pstart, pend in zip(all_pcurrent, all_pend):
        start_wbuffer = int(pstart * frequency)
        end_wbuffer = int(pend * frequency)
        for s in range(max(0,min(start_wbuffer, buffer_features[0].shape[0]-1)), min(end_wbuffer, buffer_features[0].shape[0])):
            simul_buffer[s] += [parser_features[word_id]]
            context_buffer[s] += [word_id]
            # ## Reset mechanism:
            if len(context_buffer[s]) > 1 and context[word_id][0] != context[context_buffer[s][-2]][0]:  # if the word is the first word of a sentence, we reset the buffer
                simul_buffer[s] = simul_buffer[s][-1:]  # resets and keep only the last word in the buffer
                context_buffer[s] = context_buffer[s][-1:]  # resets and keep only the last word in the buffer
        word_id += 1
    
    arr_sid = np.array([c[0] for c in context]) # shape (nb_words,)
    arr_wid = np.array([c[1] for c in context]) # shape (nb_words,)

    ## Unpops the simul_buffer to generate the buffer features
    for t in range(buffer_features[0].shape[0]):
        # if len(simul_buffer[t])>nb_buffer:
        #     warnings.warn(f"The number of elements in the buffer at time {t/frequency}s is greater than the number of subspaces. This may lead to unexpected results. In this case the activity are added into the last buffer.")
        # Parser mechanism:
        if len(simul_buffer[t])>0:
            
            ## Parser mechanism: 
            buf_idx = np.array(context_buffer[t])   # indices of words in buffer
            buf_sids = arr_sid[buf_idx]
            buf_wids = arr_wid[buf_idx]
            unique_sids, inv = np.unique(buf_sids, return_inverse=True)
            max_wids = np.full(len(unique_sids), -1, dtype=int)
            np.maximum.at(max_wids, inv, buf_wids)
            indexes = max_wids[inv].tolist()

            for b in range(min(len(simul_buffer[t]),nb_buffer)):
                buffer_features[b][t,:] = simul_buffer[t][b][...,indexes[b]]
        ## Add mechanism for writing error:
        if len(simul_buffer[t])>nb_buffer:
            for b in range(nb_buffer,len(simul_buffer[t])):
                buffer_features[-1][t,:] += simul_buffer[t][b][...,len(simul_buffer[t])-1]
    return buffer_features


class SymBufferAddParseReset_sharedreg(AbstractBuffer):
    """
        Buffer --> Reverse buffer pile
    """
    nb_buffer : int = Field(..., description="Maximal number of subpsaces in the buffer, both for the positive and negative time")
    buffer_duration: float = Field(..., description="Duration of the past buffer (Buffer) in s")
    def __call__(self,features:np.ndarray, t_current:tp.List[float],
                    context:tp.List[tp.Any])->tp.List[np.ndarray]:
        print("Generating double buffer features...")
        future_buffer_features = generate_bufferAddParseReset_features(features,t_current,self.duration,self.frequency,
                                                   self.nb_buffer,self.prediction_start,self.prediction_end,
                                                   context)
        past_buffer_features = generate_reverse_bufferAddParseReset_features(features,t_current,self.duration,self.frequency,self.nb_buffer,
                                                                self.prediction_start-self.buffer_duration,
                                                                self.prediction_end-self.buffer_duration,
                                                                context)
        return [np.concatenate(past_buffer_features + future_buffer_features, axis=1)]
