#%%
import copy
from pathlib import Path
from typing import List
import os
from alpes.buffers import SymTapAdd_sharedreg,SymBufferAdd_sharedreg
from alpes.pipeline import FeatureEncodingBufferNoKernel
import numpy as np
from scratchieegNatureNeuro2026.data_loc import proj

output_folder = proj/("featurereg_timecourseBestMix_Alpes")/"mingomni"/"mingomni_1.5"/"audio"
os.makedirs(output_folder,exist_ok=True)
output_activ = False
import pandas as pd
events = pd.read_csv(Path(__file__).parent.parent.parent/"features"/"events.csv")
words = events[events["type"]=="Word"]
t_words = words["start"].values +words["duration"].values
## add the position of the word inside the sentence:
words["position_in_sentence"] = words.groupby("sentence_id").cumcount()
phones = events[events["type"]=="Phoneme"]
t_phones = phones["start"].values +phones["duration"].values
t_elements = {"phonemeMFA": t_phones,
            "wordform": t_words,
            "lexicalSyntactic": t_words,
            "syntacticOperations": t_words,
            "syntacticState": t_words,
            "MDStree_9d": t_words,
            "MDS_CStree_9d": t_words}
contexts = {"phonemeMFA": [None]*len(t_phones),
            "wordform": words[["sentence_id","position_in_sentence"]].values.tolist(),
            "lexicalSyntactic": words[["sentence_id","position_in_sentence"]].values.tolist(),
            "syntacticOperations": words[["sentence_id","position_in_sentence"]].values.tolist(),
            "syntacticState": words[["sentence_id","position_in_sentence"]].values.tolist(),
            "MDStree_9d": words[["sentence_id","position_in_sentence"]].values.tolist(),
            "MDS_CStree_9d": words[["sentence_id","position_in_sentence"]].values.tolist()}
durations = {"phonemeMFA": 2,
             "wordform": 2,
             "lexicalSyntactic": 2,
             "syntacticOperations": 2,
             "syntacticState": 2,
             "MDStree_9d": 2,
             "MDS_CStree_9d": 2}
buffer_mapping = {f:{} for f in list(durations.keys())} #+["phoneme"]
buffer_classes = {f:None for f in list(durations.keys())} #+["phoneme"]
period = 0.05
for f in durations.keys():
    buffer_mapping[f]["phase"] = 0.0
    buffer_mapping[f]["frequency"] = 20
    buffer_mapping[f]["prediction_start"] = 0.0
    buffer_mapping[f]["prediction_end"] =  durations[f]
    buffer_mapping[f]["buffer_duration"] = durations[f]
    buffer_mapping[f]["period"] = period ## TRF
    buffer_mapping[f]["nb_buffer"] = int(np.ceil(buffer_mapping[f]["buffer_duration"]/buffer_mapping[f]["period"]))
    buffer_classes[f] = SymTapAdd_sharedreg 

#%%
dim = 9
features_names = ["phonemeMFA","wordform","lexicalSyntactic","syntacticOperations","syntacticState"][:]
features_zarr_path = [str(proj/"features_allunivariate.zarr") for _ in features_names]  # Directories or files of all features to evaluate (zarr stores)
features_keys = [name for name in features_names]  # Keys in the zarr group of the features
subject = "allelec" 
Y_zarr_path = proj/(subject+"_timecourse.zarr") #HighSR
Y_key = "train_ecog_"
os.makedirs(output_folder/f"dim_{dim}_TRFnotree",exist_ok=True)
model =  FeatureEncodingBufferNoKernel(
            subject=subject,
            output_activ=output_activ,
            features_zarr_path=features_zarr_path[:],
            features_keys=features_keys[:],
            features_names=features_names[:],
            Y_zarr_path=Y_zarr_path,
            Y_key=Y_key,
            device="cuda",
            t_elements=t_elements,
            contexts=contexts,
            decimation = lambda x: x,
            timegen=False,
            times_id = [0],
            output_folder=output_folder/f"dim_{dim}_TRFnotree",
            buffer_mapping = buffer_mapping,
            buffer_classes = buffer_classes,
            chunk_size = 100,
            max_iter=100,
            max_iter_inner_init = 50,
            )
for chunk_start in np.arange(0,model.nb_output,model.chunk_size)[1:]:
    print("==== Running chunk start: ",chunk_start)
    model.chunk_start = chunk_start
    model_tmp = copy.deepcopy(model)
    model_tmp()

# Gather all the results into a single file:
import numpy as np
import pickle
from pathlib import Path
output_folder = proj/"featurereg_timecourseBestMix_Alpes"/"mingomni"/"mingomni_1.5"/"audio"/f"dim_{dim}_TRFnotree"
all_weights = {fold: [] for fold in range(4)}
all_r2 = {fold: [] for fold in range(4)}
for chunk_start in np.arange(0,model.nb_output,step=model.chunk_size):
    for fold in range(4):
        _,weights, _, r2_split, _,_ = pickle.load(open(output_folder/f"fold_{fold}_chunk_{chunk_start}_results.pkl","rb"))
        all_weights[fold].append(weights)
        all_r2[fold].append(r2_split)
all_weights = {f:np.stack([np.concatenate([all_weights[fold][c][f] for c in range(len(all_weights[fold]))],axis=-1) for fold in range(4)],axis=0) for f in features_names}
all_r2 = {f:np.stack([np.concatenate([all_r2[fold][c][f] for c in range(len(all_r2[fold]))],axis=-1) for fold in range(4)],axis=0) for f in features_names}
pickle.dump((all_weights, all_r2), open(output_folder/f"results.pkl","wb"))

# %%
