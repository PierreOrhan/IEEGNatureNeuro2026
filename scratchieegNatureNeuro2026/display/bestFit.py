#%%
import numpy as np
from pathlib import Path
from typing import List
import submitit
import os
from itertools import product
from alpes.buffers import SymTapAdd_sharedreg,SymBufferAdd_sharedreg
from alpes.pipeline import FeatureEncodingBufferNoKernel
import submitit
from scratchieegNatureNeuro2026.data_loc import proj
from scratchieegNatureNeuro2026.display.colormaps import color_hierarchy, features_to_label
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

output_folder = proj/("featurereg_timecourseBestMix_Alpes")/"mingomni"/"mingomni_1.5"/"audio"
os.makedirs(output_folder,exist_ok=True)
output_activ = False
import pandas as pd
events = pd.read_csv(str(Path(__file__).parent.parent.parent/"features"/"events.csv"))
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
    if f in ["phonemeMFA"]: 
        buffer_classes[f] = SymTapAdd_sharedreg #_sharedreg
    else:
        buffer_classes[f] = SymBufferAdd_sharedreg #_sharedreg
        buffer_mapping[f]["nb_buffer"] = 30

    
dim = 9
features_names = ["phonemeMFA","wordform","lexicalSyntactic","syntacticOperations","syntacticState","MDStree_9d","MDS_CStree_9d"]
features_zarr_path = [str(Path(__file__).parent.parent.parent/"features"/"features_allunivariate.zarr") for _ in features_names]  # Directories or files of all features to evaluate (zarr stores)
features_keys = [name for name in features_names]  # Keys in the zarr group of the features
subject = "bestelec" 
Y_zarr_path = proj/(subject+"_timecourse.zarr") #HighSR
Y_key = "train_ecog_"
os.makedirs(output_folder/f"dim_{dim}_Local",exist_ok=True)
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
            output_folder=output_folder/f"dim_{dim}_Local",
            buffer_mapping = buffer_mapping,
            buffer_classes = buffer_classes,
            chunk_size = 258,
            max_iter=100,
            max_iter_inner_init = 50,
            )
from alpes.LSMR_pertarget import *
from alpes.pipeline.RidgeReg_subspaces_buffer_nokernel import *
import tqdm
model.load_Y()
inner_cv = KFold(n_splits=5, shuffle=False) # inner 5-fold cross-validation setup
fold = KFold(4, shuffle=False) # outer 4-fold cross-validation setup
all_indexes = list(fold.split(model.Y))
train_index,test_index = next(iter(all_indexes))
names_feature,Ktrain_feature,Ktest_feature,dim_feature = model.online_load_GPU(train_index =train_index,test_index=test_index)

Y_train,Y_test = perfold_normalization(model.Y,train_index,test_index)
Xs = [torch.tensor(ktrain,device="cuda") for ktrain in Ktrain_feature]
y = torch.tensor(Y_train,device="cuda")
alphas = torch.logspace(-3, 8, 40, dtype=Xs[0].dtype, device=Xs[0].device)
cv_splits = [
    (torch.from_numpy(tr).long(),
    torch.from_numpy(va).long())
    for tr, va in inner_cv.split(y)
]
w_syntax, log_lam_out, losses = solve_banded_ridge(
    Xs, y, cv_splits,
    n_iter_outer=model.max_iter, lr=0.01, optimizer="adam", #lbfgs
    init_alphas = alphas,
    max_iter_inner=5,
    max_iter_inner_init=model.max_iter_inner_init,
    tol_inner=list(np.geomspace(1e-2, 1e-6, model.max_iter)), verbose=True,
)
Kfeature_test = Ktest_feature
w_out_split_feature = None
w_out_split_feature = {f:w_syntax[idf].reshape(-1,dim_feature[f],w_syntax[idf].shape[-1]) for idf,f in enumerate(features_names)}
r2_full_split = r2_score_split(Y_test.to("cuda"),torch.stack([x@k for x,k in zip(Kfeature_test,w_syntax)],axis=0).to("cuda"))
r2_full_split = {f:r2_full_split[i].cpu() for i,f in enumerate(features_names)}
K_trf = [k.reshape(k.shape[0],-1,dim_feature[f]) for k,f in zip(Kfeature_test,features_names)]
subpsace_pred = []
for f,e in zip(features_names,K_trf):
    subpsace_pred += [e[:,i,:] @ w_out_split_feature[f][i] for i in range(e.shape[1])]
subpsace_pred = torch.stack(subpsace_pred,axis=0)
res_true = r2_score_split(Y_test.to("cuda"),subpsace_pred.to("cuda"))
best_voxel = np.argmax(res_true.sum(axis=0).cpu().numpy())
print("Best voxel:",best_voxel)
print("min:",np.min(res_true.sum(axis=0).cpu().numpy()),
      "max:",np.max(res_true.sum(axis=0).cpu().numpy()),
      "mean:",np.mean(res_true.sum(axis=0).cpu().numpy()))
#%%

color_hierarchy["phonemeMFA"] = color_hierarchy["phoneme"]
Y_preds_split = subpsace_pred

cm = plt.get_cmap("Spectral")
mapping = np.random.choice(len(set(words["sentence_id"])), size=len(set(words["sentence_id"])), replace=False)
fig,ax = plt.subplots(figsize=(10,6))
for ids,s in enumerate(sorted(set(words["sentence_id"]))):
    ax.vlines(words[words["sentence_id"]==s]["start"],-2,-3,color=cm(mapping[ids]/len(set(words["sentence_id"]))),alpha=1,linewidth=0.5)
ax.plot(np.arange(0,30*60,step=0.05)[test_index],Y_preds_split.sum(axis=0)[:,best_voxel].cpu().numpy(),color="red",label="predicted \n high-gamma")
ax.plot(np.arange(0,30*60,step=0.05)[test_index],Y_test[:,best_voxel].cpu().numpy(),color="black",alpha=0.4,linewidth=0.5,label="recorded \n high-gamma")
ax.set_xlim(120,140)
ax.set_xlabel("Time (s)")
for ids,s in enumerate(sorted(set(words["sentence_id"]))):
    word_s = words[(words["sentence_id"]==s) & (words["start"]>120) & (words["start"]<140)]
    last_word_y = []
    last_word_x = []
    for _,w in word_s.iterrows():
        ## Position the word a bit higher if there is overlap
        if len(last_word_y)>0 and w["start"]-last_word_x[-1] < 0.15:
            y = last_word_y[-1] + 0.3
        else:
            y = -3
        last_word_x += [w["start"]]
        last_word_y += [y]
        ax.text(w["start"],y,w["text"],rotation=90,verticalalignment="bottom",horizontalalignment="center",fontsize=7)
ax.legend(loc="upper left")
ax.set_ylim(-3.5,3)
ax.plot()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
axi = ax.inset_axes([0.8, 0.7, 0.3, 0.3])
r2 = []
for idf,f in enumerate(features_names):
    r2 += [r2_full_split[f][best_voxel]]
r2 += [1-torch.sum(torch.tensor(r2))]
axi.pie(r2, colors=[color_hierarchy[f] for f in features_names] + ["#BBB5B5D5"],startangle=0)
        # labels=[features_to_label[f] for f in features_names]+["Not explained"], 
        # autopct='%1.1f%%', startangle=0)
fig.show()
fig.savefig(Path(__file__).parent.parent.parent/"figures"/"bestFit.svg",dpi=300,format="svg",bbox_inches="tight")
# %%
