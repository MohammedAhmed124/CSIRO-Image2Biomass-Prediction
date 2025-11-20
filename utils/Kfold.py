import subprocess
from sklearn.metrics import r2_score
import numpy as np
import torch 
import pandas as pd
from sklearn.model_selection import StratifiedKFold , GroupKFold , KFold
import os
def create_folds(
    df: pd.DataFrame,
    n_splits: int,
    strategy: str = "random",
    random_state: int = 42,
    stratify_col: str = None,
    group_col: str = None,
) -> pd.Series:
    """
    Return a Series of fold ids (0..n_splits-1) aligned with df.index.

    - strategy="random": plain KFold (shuffle).
    - strategy="stratified": StratifiedKFold on stratify_col (must be categorical).
    - strategy="group": GroupKFold. If stratify_col is provided, stratify by group's mode.
    """
    folds = pd.Series(-1, index=df.index, dtype=int)
    n = len(df)

    if strategy == "random":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (_, val_idx) in enumerate(kf.split(np.arange(n))):
            folds.iloc[val_idx] = fold
        return folds

    if strategy == "stratified":
        if stratify_col is None:
            raise ValueError("stratify_col required for stratified strategy")
        if pd.api.types.is_numeric_dtype(df[stratify_col]):
            raise ValueError("stratify_col must be categorical for pure stratification")
        y = df[stratify_col].astype(str)  # ensure stable labels
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (_, val_idx) in enumerate(skf.split(np.arange(n), y)):
            folds.iloc[val_idx] = fold
        return folds

    if strategy == "group":
        if group_col is None:
            raise ValueError("group_col required for group strategy")

        groups = df[group_col].values

        if stratify_col is None:
            gkf = GroupKFold(n_splits=n_splits)
            for fold, (_, val_idx) in enumerate(gkf.split(np.arange(n), groups=groups)):
                folds.iloc[val_idx] = fold
            return folds

        # group + stratify: derive a single label per group using mode (categorical only)
        if pd.api.types.is_numeric_dtype(df[stratify_col]):
            raise ValueError("stratify_col must be categorical when used with group stratification")

        grp_series = df[[group_col, stratify_col]].groupby(group_col)[stratify_col]

        def _mode(x):
            m = x.mode()
            return str(m.iloc[0]) if len(m) else str(x.iloc[0])

        grp_label = grp_series.apply(_mode)
        unique_groups = grp_label.index.to_numpy()
        labels = grp_label.to_numpy()

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (_, val_grp_idx) in enumerate(skf.split(unique_groups, labels)):
            val_groups = set(unique_groups[val_grp_idx])
            mask = df[group_col].isin(val_groups)
            folds.loc[mask] = fold

        return folds

    raise ValueError(f"Unknown strategy: {strategy}")



def find_checkpoint_to_continue_upon(n_splits , n_epochs , output_dir):
    start_fold = None
    start_epoch = None
    checkpoint = None
    def _get_epoch_count(path):
        return int(os.path.splitext(os.path.basename(path))[0][-3:])
    for fold in range(n_splits):
        dir_path = os.path.join(output_dir, f"fold_{fold}", "checkpoints")
        dir_path_cpts = os.listdir(dir_path)
        if not dir_path_cpts:
            start_fold=fold
            start_epoch = 0
            print(f"Detected empty fold: {fold}")
            return start_fold , start_epoch , checkpoint
        files = sorted(dir_path_cpts , key = lambda path:_get_epoch_count(path) )
        if files:
            last_file = files[-1]
            file_epoch = _get_epoch_count(last_file)
            if file_epoch<n_epochs:
                print(f"Detected fold checkpoint epoch: {file_epoch} fold: {fold}")
                start_fold=fold
                start_epoch = file_epoch
                checkpoint = torch.load(
                    os.path.join(dir_path , last_file)
                )
                break
    
    return start_fold , start_epoch , checkpoint