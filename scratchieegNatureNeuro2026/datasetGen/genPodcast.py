#%%
import exca
import neuralset as ns
import numpy as np
import tqdm
import zarr as zr
import typing as tp
from pathlib import Path
from neuralfetch import register_studies_path
## Register local studies folder so neuralfetch finds Zada2025
register_studies_path(Path(__file__).parent / "IEEGNatureNeuro2026")

from scratchieegNatureNeuro2026.data_loc import proj
## Change this path to the location of the Podcast dataset on your machine

xneuro  =[]
for sub_id in ["sub-0"+str(i) for i in range(1,10)]:
    study = ns.Study(name='Zada2025',
                      path=proj,
                      infra_timelines=exca.MapInfra(folder=proj/".cache",mode="force"),
                      query="subject=='Zada2025/"+sub_id+"'")
    events = study.build()
    # Choose the right neural feature for the study (Ieeg here)
    neuro = ns.extractors.IeegExtractor(frequency=20,
                            notch_filter=60,
                            filter=(70, 200),
                            picks=("seeg",),
                            apply_hilbert=True,
                            reference="bipolar",
                            drop_bads=True,
                            event_types="Ieeg",
                            mne_cpus=8)
    segmentsTrain = ns.segments.list_segments(events, 
                                    triggers=(events.type=="Ieeg"), 
                                    start=0, duration=None)
    dsTrain = ns.SegmentDataset(extractors={"neuro": neuro}, segments=segmentsTrain)
    xneuro += [dsTrain[0].data["neuro"]]
xneuro = np.concatenate(xneuro,axis=1)
zg_out = zr.open_group(proj/"allelec_timecourse.zarr","a")
zg_out.array(name="train_ecog_0",data=xneuro.transpose(2,0,1)[:,0,:],chunks=(None,None),overwrite=True)
#%%
import zarr as zr
zgelec = zr.open_group(Path(__file__).parent.parent.parent/"features"/"coordselec.zarr","r")
coords = zgelec["coords"][:]
filter_unit = zgelec["functional_filter"][:].astype(bool)
zg_out = zr.open_group(proj/"allelec_timecourse.zarr","a")
zgbest_out = zr.open_group(proj/"bestelec_timecourse.zarr","a")
zgbest_out.array(name="train_ecog_0",data=zg_out["train_ecog_0"][:][:,filter_unit],chunks=(None,None),overwrite=True)
# %%
