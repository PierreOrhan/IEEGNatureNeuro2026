#%%
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
from pathlib import Path
from scratchieegNatureNeuro2026.display.colormaps import color_hierarchy, features_to_label
from scratchieegNatureNeuro2026.data_loc import proj
features_to_label["phonemeMFA"] = features_to_label["phoneme"]

output_folder = proj/("featurereg_timecourseBestMix_Alpes_Marr")/"mingomni"/"mingomni_1.5"/"audio"/"dim_9_Mix"
all_weights_mix, all_r2_mix = pickle.load(open(output_folder/f"results.pkl","rb"))
output_folder = proj/("featurereg_timecourseBestMix_Alpes_Marr")/"mingomni"/"mingomni_1.5"/"audio"/"dim_9_TRFnotree"
all_weights_trfnotree, all_r2_trfnotree = pickle.load(open(output_folder/f"results.pkl","rb"))
output_folder = proj/("featurereg_timecourseBestMix_Alpes_Marr")/"mingomni"/"mingomni_1.5"/"audio"/"dim_9_TRF"
all_weights_trf, all_r2_trf = pickle.load(open(output_folder/f"results.pkl","rb"))
output_folder = proj/("featurereg_timecourseBestMix_Alpes")/"mingomni"/"mingomni_1.5"/"audio"/"dim_9_Buffer"
all_weights_buffer, all_r2_buffer = pickle.load(open(output_folder/f"results.pkl","rb"))

#%%
import numpy as np
import matplotlib.pyplot as plt
features_names = ["phonemeMFA","wordform","lexicalSyntactic","syntacticOperations","syntacticState","MDStree_9d","MDS_CStree_9d"]
def get_xtot(all_r2,features_names):
    r2_feature = []
    for f in features_names:
        # all_r2[f][all_r2[f]<0] = 0
        r2_feature.append(all_r2[f].sum(axis=1).mean(axis=0))
    r2_feature = np.stack(r2_feature,axis=0)
    r2_ratio = r2_feature[:]
    r2_ratio[r2_ratio<0] = 0
    r2_ratio = r2_ratio/np.sum(r2_ratio,axis=0,keepdims=True)
    xtot = r2_feature.sum(axis=0)
    return xtot,r2_ratio,r2_feature
xtot_buffer,r2_ratio_buffer,r2_feature_buffer = get_xtot(all_r2_buffer,features_names)
xtot_trf,r2_ratio_trf,r2_feature_trf = get_xtot(all_r2_trf,features_names)
xtot_trfnotree,r2_ratio_trfnotree,r2_feature_trfnotree = get_xtot(all_r2_trfnotree,features_names[:-2])
xtot_mix,r2_ratio_mix,r2_feature_mix = get_xtot(all_r2_mix,features_names)

#%%
import zarr as zr
zgelec = zr.open_group(Path(__file__).parent.parent.parent/"features"/"coordselec.zarr","r")
coords = zgelec["coords"][:]
filter_unit = zgelec["functional_filter"][:].astype(bool)

#%%
print(np.nanmean((xtot_trf[filter_unit]-xtot_trfnotree[filter_unit])))
print(np.nanmax((xtot_trf[filter_unit]-xtot_trfnotree[filter_unit])))
print(np.nanmin((xtot_trf[filter_unit]-xtot_trfnotree[filter_unit])))
import scipy.stats as scs
scs.ttest_rel(xtot_trf[filter_unit],xtot_trfnotree[filter_unit],alternative="greater")

#%%
to_plot = [xtot_trfnotree,xtot_trf,xtot_buffer,xtot_mix]
fig,ax = plt.subplots(figsize=(4,3))
ax.bar(range(len(to_plot)),[x[filter_unit].mean() for x in to_plot],
       yerr=[x[filter_unit].std()/np.sqrt(x[filter_unit].shape[-1]) for x in to_plot],
       width=0.4,color="grey")
ax.set_ylabel("Predicted Variance (R²)")
ax.set_xticks(range(len(to_plot)))
ax.set_xticklabels(["TRF model","TRF model \n with Tree-coding features","Buffer model \n with Tree-coding features","TRF (Phoneme) \n Buffer model (Others) \n with Tree-coding features"],rotation=90,ha="center")
# remove spines:
import scipy.stats as stats
for i in range(len(to_plot)-1):
    t_stat,p_val = stats.ttest_rel(to_plot[i][...,filter_unit],to_plot[i+1][...,filter_unit],axis=-1,alternative="less")
    if p_val<0.001:
        ax.text(i+0.35,np.mean(to_plot[i+1][...,filter_unit])+0.0025,"***",color="k")
        ax.hlines(np.mean(to_plot[i+1][...,filter_unit])+0.002,i,i+1,color="k",linestyle="-",linewidth=0.5)
[ax.spines[spine].set_visible(False) for spine in ["top","right"]]
fig.show()
fig.savefig(Path(__file__).parent.parent.parent/"figures"/"modelComparison.svg",dpi=300,format="svg",bbox_inches="tight")

# %%
