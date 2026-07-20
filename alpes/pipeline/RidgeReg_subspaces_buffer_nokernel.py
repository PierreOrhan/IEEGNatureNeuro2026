from pathlib import Path
from pydantic import Field, model_validator
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import zarr as zr
import os
import copy
import gc
import tqdm
from typing import Optional,List,Any
from alpes.buffers import AbstractBuffer
from .lazy import LazyFeatureRidgeEncoding
import pickle
import exca as xk
from alpes import solve_banded_ridge, compute_gradient, solve_primal, BandedRidgeOperator



def r2_score_split(y_true, y_pred, include_correlation=True):
    """Split the R2 score into individual components using the product measure.

    When estimating a linear joint model, the predictions of each feature space
    are summed::

        Yhat_joint = Yhat_A + Yhat_B + ... + Yhat_Z

    The joint model R2 can be computed as::

        R2_joint = R2(Yhat_joint, Y)

    This function estimates the contribution of each feature space to the joint
    model R2 such that::

        R2_joint = R2_A + R2_B + ... + R2_Z

    Mathematically, this is achieved by taking into account the correlations
    between predictions (i.e. Yhat_A*Yhat_B,..., Yhat_A*Yhat_Z). The function
    can also returns an estimate that ignores these correlations.

    This function differs from r2_score_split_svd in the method used to
    decompose the variance. The function r2_score_split is based on the product
    measure method, while the function r2_score_split_svd is based on the
    relative weights method.

    This function assumes that y_true is zero-mean over samples.

    Parameters
    ----------
    y_true : array of shape (n_samples, n_targets)
        Observed data. Has to be zero-mean over samples.
    y_pred : array of shape (n_kernels, n_samples, n_targets) 
        Predictions.
    include_correlation : bool
        Whether to include correlation between feature spaces.
        If True, individual feature space R2 sum is equivalent to the joint
        model R2 (i.e. from `y_pred.sum(0)`).

    Returns
    -------
    r2 : array (n_kernels, n_targets) or (n_targets, )
        Individual feature space R2 scores.
    """
    sst = (y_true ** 2).sum(0)  # (n_targets,)
    no_split = y_pred.ndim == 2
    if no_split:
        y_pred = y_pred[None]

    # inter[k, t] = sum_s y_pred[k,s,t] * y_true[s,t]
    inter_batch = torch.einsum('kst,st->kt', y_pred, y_true)

    if include_correlation:
        # asst[k, t] = sum_s y_pred[k,s,t] * sum_{k'} y_pred[k',s,t]
        asst_batch = torch.einsum('kst,st->kt', y_pred, y_pred.sum(0))
    else:
        asst_batch = (y_pred ** 2).sum(1)

    r2 = (2 * inter_batch - asst_batch) / sst[None]

    if no_split:
        r2 = r2[0]

    return r2



def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output

def perfold_normalization(Y,train_index,test_index):
    scaler = StandardScaler()
    Y_train = torch.tensor(scaler.fit_transform(Y[train_index]),dtype=torch.float32)
    Y_test = torch.tensor(scaler.transform(Y[test_index]),dtype=torch.float32)
    ## Poor mean estimation fix: we substract any remaining mean to avoid issues when using R2_score_split:
    Y_test = Y_test-torch.mean(Y_test,dim=0,keepdim=True)
    return Y_train,Y_test

def online_buffer_gpu_nokernel(features_zarr_path, features_names, features_keys, buffer_mapping, buffer_classes, t_elements, train_index, test_index, decimation=None,contexts=None):
    """
        Load the input features and apply the buffers.
        Note: Na time points/elements after buffering are set to the 0 after mean and std normalization.
        Therefore they do not contribute to the predictions at all.

        features_zarr_path: list of paths to the zarr stores of the features, shape: (nb_events,dimension) or (nb_events,dimension,context_dimension)
        features_names: list of names of the features
    """
    ## Load features:
    K_features = []
    dim_feature = {}
    for fpath,fname,fk in zip(features_zarr_path,features_names,features_keys):
        
        if buffer_classes is not None and fname in buffer_classes.keys():
            buffer = buffer_classes[fname](**buffer_mapping[fname])   
            if contexts is None:
                subspaces = buffer(features=zr.open_group(fpath,"r")[fk][:,:], 
                                t_current=t_elements[fname], 
                                context=None) 
            else:
                subspaces = buffer(features=zr.open_group(fpath,"r")[fk][:,:], 
                                t_current=t_elements[fname], 
                                context=contexts[fname]) 
        else:
            subspaces = [zr.open_group(fpath,"r")[fk][:,:]]
        dim_feature[fname] = zr.open_group(fpath,"r")[fk].shape[1]

        for idx,x in enumerate(subspaces):
            if buffer_classes is not None and fname in buffer_classes.keys():
                fkey_time = str(fname)+f"_{buffer_classes[fname].__name__}"+str(idx)
            else:
                fkey_time = str(fname)+"_"+str(idx)
            
            xfeature = torch.tensor(x,dtype=torch.float32).to("cuda")
            mu = torch.nanmean(xfeature[train_index,:],dim=0,keepdim=True)
            std  = nanstd(xfeature[train_index,:],dim=0,keepdim=True)
            # xfeature = (xfeature - mu) / std
            xfeature[:,std[0,:]!=0] = ((xfeature - mu) / std)[:,std[0,:]!=0]
            xfeature[:,std[0,:]==0] = 0

            xfeature[torch.isnan(xfeature)] = 0 # Put the NAN as 0, so that they don't contribute to these predictions!
            xfeature  = decimation(xfeature) if decimation is not None else xfeature ## Decimates the features.

            xf_train = xfeature[train_index,:]
            xf_test = xfeature[test_index,:]
            K_features += [(fkey_time,xf_train,xf_test)]
            del xfeature
            torch.cuda.empty_cache()
    Ktrain_feature = [Kf[1] for Kf in K_features]
    Ktest_feature = [Kf[2] for Kf in K_features]
    names_feature = [Kf[0] for Kf in K_features]
    del K_features

    return names_feature,Ktrain_feature,Ktest_feature,dim_feature


def load_Y(Y_zarr_path,Y_key,nb_output,times_id=None,chunk_size=-1,chunk_start=0):
    zg = zr.open_group(Y_zarr_path,mode="r")
    if "functional_filter" in zg.keys():
        filter_unit = np.where(zg["functional_filter"][:].astype(bool))[0]

    if chunk_size == -1:
        chunk_size = zg[Y_key+str(times_id[0])].shape[1]
    Y = []
    if times_id is None:
        times_id = np.sort([int(e.replace(Y_key,"")) for e in list(filter(lambda e:e.startswith(Y_key), zg.keys()))])
    for time_id in times_id:
        if "functional_filter" in zg.keys():
            res = zg[Y_key+str(time_id)][:,filter_unit][:,chunk_start:chunk_start+chunk_size]
        else:
            res = zg[Y_key+str(time_id)][:,chunk_start:chunk_start+chunk_size]
        Y += [torch.tensor(res,dtype=torch.float32)]
    Y = torch.concat(Y,dim=1)
    
    if "functional_filter" in zg.keys():
        voxel_ids = filter_unit[chunk_start:min(chunk_start+chunk_size,nb_output)]
    else:
        voxel_ids = np.arange(chunk_start,min(chunk_start+chunk_size,nb_output))
        
    times = np.stack([times_id for _ in range(voxel_ids.shape[-1])],axis=1).reshape(-1)
    return Y,voxel_ids,times,times_id

def get_r2(w_out,Kfeature_test,Y_test,features_names,dim_feature : dict[str]):
    """
        Compute the R2 score for each feature and each subspace.
    """
    w_out_split_feature = None
    w_out_split_feature = {f:w_out[idf].reshape(-1,dim_feature[f],w_out[idf].shape[-1]) for idf,f in enumerate(features_names)}
    
    r2_full_split = r2_score_split(Y_test.to("cuda"),torch.stack([x@k for x,k in zip(Kfeature_test,w_out)],axis=0).to("cuda"))
    r2_full_split = {f:r2_full_split[i].cpu() for i,f in enumerate(features_names)}

    K_trf = [k.reshape(k.shape[0],-1,dim_feature[f]) for k,f in zip(Kfeature_test,features_names)]
    subpsace_pred = []
    for f,e in zip(features_names,K_trf):
        subpsace_pred += [e[:,i,:] @ w_out_split_feature[f][i] for i in range(e.shape[1])]
    subpsace_pred = torch.stack(subpsace_pred,axis=0)
    r2_trf = r2_score_split(Y_test.to("cuda"),subpsace_pred.to("cuda"))
    ## reorganize r2_trf into a dictionary of features:
    clen_split = np.cumsum([0]+[e.shape[1] for e in K_trf])
    r2_trf_split = {f:r2_trf[clen_split[i]:clen_split[i+1]].cpu() for i,f in enumerate(features_names)}
    w_out_split_feature = {f:w_out_split_feature[f].cpu() for f in features_names}
    return w_out_split_feature,r2_full_split,r2_trf_split

class FeatureEncodingBufferNoKernel(LazyFeatureRidgeEncoding):
    """
        Feature encoding with ridge regression and subspaces, using buffers or TRF for the features.
    """
    timegen: bool = False
    buffer_mapping: Optional[dict] = None # dictionary of kwargs to input inside the buffer creator.
    buffer_classes: Optional[dict[str, Any]] = None #type of buffer to use for each subspace.
    t_elements : Optional[dict[str, List]] = None # Time of all elements to be encoded.
    contexts: Optional[dict[str, List]] = None # Contextual information for each element in the dataset
    decimation: Optional[callable] = None # decimation function to apply to the features (e.g., decimate by 2, by 4, etc.)
    chunk_start : int = 0
    max_iter_inner_init: int = 50

    # infra: xk.TaskInfra = xk.TaskInfra()

    @model_validator(mode="after")
    def _indicateDurationToBuffer(self):
        """
            Indicates the total duration of the stimuli to the buffer, so that it can generate the right number of features.
        """
        zg = zr.open_group(self.Y_zarr_path,mode="r")
        if (self.buffer_mapping is not None) and len(self.buffer_mapping.keys())>0:
            assert np.all([self.buffer_mapping[f]["frequency"]==self.buffer_mapping[self.features_names[0]]["frequency"] for f in self.features_names if f in self.buffer_mapping.keys()]), "The frequency of the buffer should be the same for all features"
            self.duration = zg[self.Y_key+str(0)].shape[0]/self.buffer_mapping[self.features_names[0]]["frequency"] 
            for f in self.buffer_mapping.keys():
                self.buffer_mapping[f]["duration"] = self.duration
        return self

    def load_Y(self):
        self.Y,self.voxel_ids,self.times,self.times_id = load_Y(self.Y_zarr_path,self.Y_key,
                                                                self.nb_output,self.times_id,
                                                                chunk_size=self.chunk_size,
                                                                chunk_start=self.chunk_start)
        self.Y = self.decimation(self.Y) if self.decimation is not None else self.Y
        
        self.times = np.stack([self.times_id for _ in range(self.voxel_ids.shape[-1])],axis=1).reshape(-1)
        self.voxel_id = copy.deepcopy(self.voxel_ids)
        self.voxel_ids = np.stack([self.voxel_ids for _ in range(len(self.times_id))],axis=0).reshape(-1)


    def online_load_GPU(self,train_index,test_index):
        """
            Load the input features and apply the buffers.
        """
        names_feature,Ktrain_feature,Ktest_feature,dim_feature = online_buffer_gpu_nokernel(features_zarr_path=self.features_zarr_path, 
                                                                                            features_names=self.features_names, 
                                    features_keys=self.features_keys, 
                                    buffer_mapping=self.buffer_mapping, 
                                    buffer_classes=self.buffer_classes, 
                                    t_elements=self.t_elements, train_index=train_index, 
                                    test_index=test_index, decimation=self.decimation,
                                    contexts=self.contexts)

        return names_feature,Ktrain_feature,Ktest_feature,dim_feature


    def __call__(self):
        ## Load the predicted data (Y) and the input features, apply the buffers
        self.load_Y()
        inner_cv = KFold(n_splits=5, shuffle=False) # inner 5-fold cross-validation setup
        fold = KFold(4, shuffle=False) # outer 4-fold cross-validation setup
        all_indexes = list(fold.split(self.Y))

        for fold_id, (train_index,test_index) in tqdm.tqdm(enumerate(all_indexes)):
            names_feature,Ktrain_feature,Ktest_feature,dim_feature = self.online_load_GPU(train_index =train_index,test_index=test_index)

            Y_train,Y_test = perfold_normalization(self.Y,train_index,test_index)
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
                n_iter_outer=self.max_iter, lr=0.01, optimizer="adam", #lbfgs
                init_alphas = alphas,
                max_iter_inner=5,
                max_iter_inner_init=self.max_iter_inner_init,
                tol_inner=list(np.geomspace(1e-2, 1e-6, self.max_iter)), verbose=True,
            )
            print("Estimating per feature space x (time-events locked lags) R2 score...")
            w_feature_syntax,r2_syntax,r2_trf_split = get_r2(w_syntax,Ktest_feature,Y_test,self.features_names,dim_feature)
            print("Ended estimating per feature space x (time-events locked lags) R2 score...")
            log_lam_out = log_lam_out.cpu()
            losses = torch.tensor(losses).cpu()
            
            pickle.dump((w_syntax,w_feature_syntax,r2_syntax,r2_trf_split,log_lam_out,losses), open(self.output_folder/f"fold_{fold_id}_chunk_{self.chunk_start}_results.pkl","wb"))
            # yield w_feature_syntax,r2_syntax,r2_trf_split,log_lam_out,losses


    def process(self) -> float:
        """
            Process the feature encoding with ridge regression and subspaces, using buffers or TRF for the features.
        """

        results = list(self())
        return results
