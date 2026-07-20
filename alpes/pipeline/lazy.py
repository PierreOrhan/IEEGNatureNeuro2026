from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pathlib import Path
from typing import List, Tuple, Optional,Union
import numpy as np
import zarr as zr
import os
import re

class LazyFeatureRidgeEncoding(BaseModel):
    """
    Ridge encoding from:
        model subspaces (probe_zarr_path, probe_names, probe_keys) used to project from the features (X_zarr_path, X_keys)
        univariate features(features_zarr_path, features_names, features_keys)
        to predict the activity (Y_zarr_path, Y_key) with a ridge regression with cross-validation on the regularization parameter alpha (alpha_bounds, n_alphas).

        We parse all the features an probes as List of List to allow for multiple input layers.
    """

    # Inputs
    output_folder: Path
    subject: str

    output_activ : bool = False

    features_zarr_path: List[Union[str, Path]]  # Directories or files of all features to evaluate (zarr stores)
    features_names: List[str]  # Names of the features
    features_keys: List[str]  # Keys in the zarr group of the features

    Y_zarr_path: Path  # Path to the zarr group of the predicted data
    Y_key: str  # Key in the zarr group of the predicted data
    
    alpha_bounds: Tuple[int, int] = (1, 10)
    n_alphas: int = 40

    # Chunking for Y to manage memory
    chunk_size: int = -1  # chunk size
    nb_output: int = -1  # number of units in predicted data (computed if -1)

    device : str = "cpu"
    times_id : Optional[List[int]] = None

    # Derived/runtime attributes (set after validation)
    alphas: np.ndarray = Field(default_factory=lambda: np.array([], dtype=np.float32))

    max_iter: int = 100

    # # Allow setting additional attributes at runtime (e.g., tensors, caches)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # ---- Validators ----
    @field_validator("output_folder", "Y_zarr_path")
    @classmethod
    def _path_must_exist(cls, v: Path) -> Path:
        p = Path(v)
        if not p.exists():
            raise ValueError(f"Path does not exist: {p}")
        return p

    
    @field_validator("features_zarr_path")
    @classmethod
    def _each_feature_path_exists(cls, v: List[str]) -> List[Path]:
        checked: List[Path] = []
        for it in v:
            p = Path(it)
            if not p.exists():
                raise ValueError(f"Feature path does not exist: {p}")
            checked.append(p)
        return checked


    @field_validator("alpha_bounds")
    @classmethod
    def _alpha_bounds_valid(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        a, b = v
        if a >= b:
            raise ValueError("alpha_bounds must be (low, high) with low < high")
        return v

    @model_validator(mode="after")
    def _lists_alignment_and_compute_derived(self):
        # Validate probe lists are aligned

        # Compute alphas
        a_low, a_high = self.alpha_bounds
        n_alphas = int(self.n_alphas)
        self.alphas = np.array(np.logspace(a_low, a_high, n_alphas), dtype=np.float32)

        zg = zr.open_group(self.Y_zarr_path,mode="r")
        if "functional_filter" in zg.keys():
            filter_unit = zg["functional_filter"][:]
            self.nb_output = np.sum(filter_unit)
        else:
            self.nb_output = zg[self.Y_key+str(0)].shape[1]

        os.makedirs(self.output_folder,exist_ok=True)

        # zgout = zr.open_group(self.output_folder/"encodingPredictions.zarr",mode="a")
        
        feature_names = set([fs for fs in self.features_names])

        times_id = np.sort([int(e.replace(self.Y_key,"")) for e in list(filter(lambda e:re.match(rf"^{re.escape(self.Y_key)}\d+$", e) is not None, zg.keys()))])
        self.times_id_tot = times_id
        # if self.output_activ:
        #     try:
        #         zgout.create(name="target_"+self.subject+"_"+self.Y_key,
        #                     shape=(len(times_id),)+zg[self.Y_key+str(times_id[0])].shape,
        #                     chunks=(1,None,None))
        #         zgout.create(name="pred_"+self.subject+"_"+self.Y_key,
        #                     shape=(len(times_id),)+zg[self.Y_key+str(times_id[0])].shape,
        #                     chunks=(1,None,None))
        #         for f in feature_names:
        #             delays = self.word_delays if  not (str(f) in self.exception_delays) else [0]
        #             for word_delay in delays:
        #                 fname_delayed = f"{f}_wd{word_delay}"
        #                 zgout.create(name="pred_feature_"+self.subject+"_"+self.Y_key+fname_delayed,
        #                         shape=(len(times_id),)+zg[self.Y_key+str(times_id[0])].shape,
        #                             chunks=(1,None,None))
        #     except:
        #         pass

        # Keep nb_output as provided; if negative it will be inferred lazily in load_Y()
        return self

