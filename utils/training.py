from sklearn.metrics import r2_score
import numpy as np
import torch
from typing import Callable, Optional, Dict, Any, Tuple, List
import os
from .config import CFG
import pandas as pd



def calculate_weighted_r2(
        model,
        dataloader,
        device,
        ):
    """Compute a weighted R² score and mean MSE across five biomass targets.


    This function evaluates a model over a validation dataloader and returns a
    weighted R² (computed on log-transformed values) together with the mean MSE
    computed on raw values.


    Main assumptions
    ----------------
    - The model returns three tensors in order: `(pred_total, pred_gdm, pred_green)` when called as `model(left_imgs, right_imgs)`.
    - Each batch yielded by `dataloader` is a `dict` with keys: `'left_image'`, `'right_image'` (tensors shaped `(B, C, H, W)`) and `'labels'` (tensor shaped `(B, 3)` with columns `[Total, GDM, Green]`).
    - Derived components are computed as `Dry_Clover_g = GDM - Green` and `Dry_Dead_g = Total - GDM`; predicted values for these components are clamped to >= 0.
    - R² is calculated on log-transformed values using `np.log1p(value + CFG.EPS)` for numerical stability; MSE is computed on raw (non-logged) values.
    - The configuration object (`CFG`) provides `ALL_TARGETS`, `R2_WEIGHTS`, and `EPS`, and these must be correctly defined.
    - Inputs and model tensors are moved to the supplied `device`; all NumPy conversions happen on CPU.


    Parameters
    ----------
    model : torch.nn.Module
    Model eval mode expected. Called as ``model(left_imgs, right_imgs)`` and
    must return (pred_total, pred_gdm, pred_green) as tensors.


    dataloader : Iterable[dict]
    Yields batches where each batch is a dict with keys:
    - 'left_image' : torch.Tensor, shape (B, C, H, W)
    - 'right_image': torch.Tensor, shape (B, C, H, W)
    - 'labels' : torch.Tensor, shape (B, 3) with columns [Total, GDM, Green]


    device : torch.device
    Device to move inputs to before calling the model.


    Returns
    -------
    (final_score, val_mse, r2_scores) : (float, float, dict)
    - final_score (float): weighted R² across the five targets.
    - val_mse (float): average MSE across the five targets (raw-space).
    - r2_scores (dict): per-target R² mapping target_name -> float.


    Example
    -------
    >>> device = torch.device('cuda')
    >>> final_score, val_mse, r2_scores = calculate_weighted_r2(model, val_loader, device)


    """
    # Put model in evaluation mode and disable grad rules
    model.eval()
    all_preds_dict = {target: [] for target in CFG.ALL_TARGETS}
    all_true_dict = {target: [] for target in CFG.ALL_TARGETS}

    with torch.no_grad():
        for batch in dataloader:

            left_imgs = batch['left_image'].to(device)
            right_imgs = batch['right_image'].to(device)

            # extract the three ground-truth label columns (Total, GDM, Green)
            true_total = batch['labels'][:, 0].cpu().numpy()
            true_gdm = batch['labels'][:, 1].cpu().numpy()
            true_green = batch['labels'][:, 2].cpu().numpy()

            # model predicts three outputs: Total, GDM, Green
            pred_total, pred_gdm, pred_green = model(left_imgs, right_imgs)


            pred_total = pred_total.squeeze().cpu().numpy()
            pred_gdm = pred_gdm.squeeze().cpu().numpy()
            pred_green = pred_green.squeeze().cpu().numpy()


            # clover = GDM - Green
            # dead   = Total - GDM
            true_clover = true_gdm - true_green
            pred_clover = pred_gdm - pred_green

            true_dead = true_total - true_gdm
            pred_dead = pred_total - pred_gdm

            # enforce non-negative predicted components (biomass can't be negative)
            pred_clover = np.maximum(0, pred_clover)
            pred_dead = np.maximum(0, pred_dead)

            # collect the five targets in the same key order as CFG.ALL_TARGETS
            batch_preds = {
                "Dry_Total_g": pred_total,
                "GDM_g": pred_gdm,
                "Dry_Green_g": pred_green,
                "Dry_Clover_g": pred_clover,
                "Dry_Dead_g": pred_dead
            }
            batch_true = {
                "Dry_Total_g": true_total,
                "GDM_g": true_gdm,
                "Dry_Green_g": true_green,
                "Dry_Clover_g": true_clover,
                "Dry_Dead_g": true_dead
            }


            for target in CFG.ALL_TARGETS:
                all_preds_dict[target].extend(batch_preds[target])
                all_true_dict[target].extend(batch_true[target])


    final_score = 0
    r2_scores = {}
    mse_list = []

    for target in CFG.ALL_TARGETS:
        true_vals = np.array(all_true_dict[target])
        pred_vals = np.array(all_preds_dict[target])

        true_log = np.log1p(np.maximum(0 , true_vals)+CFG.EPS)
        pred_log = np.log1p(np.maximum(0 , pred_vals)+CFG.EPS)


        score = r2_score(true_log, pred_log)
        r2_scores[target] = score
        final_score += score * CFG.R2_WEIGHTS[target]


        mse_list.append(np.mean((pred_vals - true_vals) ** 2))


    val_mse = float(np.mean(mse_list))
    return float(final_score), val_mse , r2_scores

def train_one_epoch(
        model,
        dataloader,
        optimizer,
        criterion,
        device,
        scaler,
        accumulation_steps,
        epoch,
        total_epochs
        ):

    model.train()
    total_loss = 0
    optimizer.zero_grad()


    all_preds_dict = {target: [] for target in CFG.ALL_TARGETS}
    all_true_dict = {target: [] for target in CFG.ALL_TARGETS}

    total_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        # move inputs and labels to device
        left_imgs = batch['left_image'].to(device)
        right_imgs = batch['right_image'].to(device)
        true_labels = batch['labels'].to(device)

        # split true labels into the three targets the model predicts
        true_total = true_labels[:, 0].unsqueeze(-1)
        true_gdm = true_labels[:, 1].unsqueeze(-1)
        true_green = true_labels[:, 2].unsqueeze(-1)

        # forward pass under automatic mixed precision
        with torch.amp.autocast("cuda", enabled=(device.type == 'cuda')):
            
            pred_total, pred_gdm, pred_green = model(left_imgs, right_imgs)

            # compute the three per-target losses and combine
            loss_total = criterion(pred_total, true_total)
            loss_gdm = criterion(pred_gdm, true_gdm)
            loss_green = criterion(pred_green, true_green)
            loss = loss_total + loss_gdm + loss_green

            # scale loss for gradient accumulation
            loss_for_backward = loss / accumulation_steps

        # backward pass with GradScaler for mixed precision
        scaler.scale(loss_for_backward).backward()

        # update cumulative raw loss for reporting (un-normalized by accumulation)
        total_loss += loss.item()

        # extract numpy arrays for predictions and true labels (for running metrics)
        pred_total_np = pred_total.squeeze().detach().cpu().numpy()
        pred_gdm_np = pred_gdm.squeeze().detach().cpu().numpy()
        pred_green_np = pred_green.squeeze().detach().cpu().numpy()

        true_total_np = true_total.squeeze().detach().cpu().numpy()
        true_gdm_np = true_gdm.squeeze().detach().cpu().numpy()
        true_green_np = true_green.squeeze().detach().cpu().numpy()

        # derive remaining two components on the training-side (same formulas as validation)
        true_clover = true_gdm_np - true_green_np
        pred_clover = pred_gdm_np - pred_green_np

        true_dead = true_total_np - true_gdm_np
        pred_dead = pred_total_np - pred_gdm_np

        pred_clover = np.maximum(0, pred_clover)
        pred_dead = np.maximum(0, pred_dead)

        # append this batch's results to the epoch accumulators
        batch_preds = {
            "Dry_Total_g": pred_total_np,
            "GDM_g": pred_gdm_np,
            "Dry_Green_g": pred_green_np,
            "Dry_Clover_g": pred_clover,
            "Dry_Dead_g": pred_dead
        }
        batch_true = {
            "Dry_Total_g": true_total_np,
            "GDM_g": true_gdm_np,
            "Dry_Green_g": true_green_np,
            "Dry_Clover_g": true_clover,
            "Dry_Dead_g": true_dead
        }
        for target in CFG.ALL_TARGETS:
            all_preds_dict[target].extend(np.atleast_1d(batch_preds[target]).ravel())
            all_true_dict[target].extend(np.atleast_1d(batch_true[target]).ravel())

        # optimizer step when accumulation boundary reached (or at final batch)
        if (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # compute running train metrics across processed batches (overridden each batch)
        mse_list = []
        r2_acc = 0.0
        for target in CFG.ALL_TARGETS:
            true_vals = np.array(all_true_dict[target])
            pred_vals = np.array(all_preds_dict[target])
            mse_list.append(np.mean((pred_vals - true_vals) ** 2))
            true_log = np.log1p(np.maximum(true_vals , 0)+CFG.EPS)
            pred_log = np.log1p(np.maximum(pred_vals , 0)+CFG.EPS)
            r2_acc += r2_score(true_log, pred_log) * CFG.R2_WEIGHTS[target]

        current_train_mse = float(np.mean(mse_list))
        current_train_r2 = float(r2_acc)

        # print a single-line, overridable progress line with batch index
        print(f"[{epoch + 1} / {total_epochs}] [{i + 1} / {total_batches}] |  Train Mse : {current_train_mse:.4f} | Train R2 : {current_train_r2:.4f} | Val Mse : - | Val R2 : - |  -", end="\r", flush=True)

    # compute final epoch-level train metrics from accumulators
    per_target_r2 = {}
    final_mse_list = []
    final_r2_acc = 0.0
    for target in CFG.ALL_TARGETS:
        true_vals = np.array(all_true_dict[target])
        pred_vals = np.array(all_preds_dict[target])
        final_mse_list.append(np.mean((pred_vals - true_vals) ** 2))
        true_log = np.log1p(np.maximum(true_vals , 0)+CFG.EPS)
        pred_log = np.log1p(np.maximum(pred_vals , 0)+CFG.EPS)
        target_r2 = r2_score(true_log, pred_log)
        per_target_r2[target] = target_r2
        final_r2_acc += target_r2 * CFG.R2_WEIGHTS[target]


    train_mse = float(np.mean(final_mse_list))
    train_r2 = float(final_r2_acc)

    # return average loss (per mini-batch), and epoch-level train metrics
    return total_loss / len(dataloader), train_mse, train_r2 , per_target_r2

from sklearn.metrics import r2_score
import numpy as np
import torch 

from .tensorboard import luanch_tensorboard , split_df_into_groups , df_to_image
from .Kfold import find_checkpoint_to_continue_upon , create_folds
import random
from torch.utils.data import DataLoader

def run_kfold_training(
        model_constructor: Callable[..., torch.nn.Module],
        df: pd.DataFrame,
        training_config:dict,
        BiomassDatasetClass: Callable,
        get_transforms: Callable,
        transforms_config : dict,
        kfold_params : dict = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        ds_kwargs: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
        output_dir: str = "kf_runs",
        device: Optional[torch.device] = None,
        save_every: int = 5,
        num_workers: int = 8,
        seed: int = 42,
        continue_:bool = False
    ):
    """
    K-fold trainer. Uses a SINGLE shared TensorBoard writer (global_writer).
    Logs per-fold scalars under keys namespaced with `fold_{i}/...`.
    Produces a CSV + markdown table of per-fold × per-target R2 scores in output_dir.
    """
    if device is None:
        device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    if kfold_params is None:
        kfold_params = {}

    targets = CFG.ALL_TARGETS
    batch_size = training_config["batch_size"]
    n_epochs = training_config["n_epochs"]
    accumulation_steps = training_config["accumulation_steps"]
    csv_path = os.path.join(output_dir, "per_fold_r2_table.csv")

    os.makedirs(output_dir, exist_ok=True)
    if continue_:
        start_fold , start_epoch , checkpoint = find_checkpoint_to_continue_upon(n_splits , n_epochs , output_dir)
        df_table = pd.read_csv(csv_path)
    else:
        start_fold , start_epoch , checkpoint = 0, 0 , None
        df_cols = [f"fold_{fold}_{target}" for target in targets for fold in range(1 , n_splits+1)] + [f"total_r2_fold_{fold}" for fold in range(1 , n_splits+1)]
        df_table = pd.DataFrame(columns=df_cols)


    global_writer = luanch_tensorboard(
        output_dir=output_dir,
        continue_=continue_,
        purge_step=start_epoch
        )


    folds_series = create_folds(
        df,
        n_splits=n_splits,
        **kfold_params
        )
    df2 = df.copy()
    df2["fold"] = folds_series





    for fold in range(start_fold , n_splits):
        print(f"\n=== Fold {fold+1} / {n_splits} ===")
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        train_df = df2[df2["fold"] != fold].reset_index(drop=True)
        val_df = df2[df2["fold"] == fold].reset_index(drop=True)


        fold_seed = seed + fold
        np.random.seed(fold_seed); random.seed(fold_seed); torch.manual_seed(fold_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(fold_seed)

        train_tfm, val_tfm = get_transforms(
            **transforms_config
            ) #augmentation
        train_ds = BiomassDatasetClass(
            train_df,
            CFG.IMAGE_DIR,
            train_tfm,
            CFG.MODEL_TARGETS,
            **(ds_kwargs or {})
            )
        val_ds = BiomassDatasetClass(
            val_df,
            CFG.IMAGE_DIR,
            val_tfm,
            CFG.MODEL_TARGETS,
            **(ds_kwargs or {})
            )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            # pin_memory=True,
            # prefetch_factor=4
            )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers//2)
            )



        if checkpoint is not None:
            print(f"Resuming from fold {start_fold}, epoch {start_epoch}")

            model = model_constructor(**(model_kwargs or {})).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

            optimizer = checkpoint["training_config"]["optimizer"](model.parameters(), **checkpoint["training_config"]["optimizer_config"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            scaler = checkpoint["training_config"]["scaler"](**checkpoint["training_config"]["scaler_config"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

            best_score = checkpoint.get("best_score", -1e9)
            fold_seed = checkpoint.get("fold_seed", seed + start_fold)
        else:
            # Normal init
            model = model_constructor(**(model_kwargs or {})).to(device)
            optimizer = training_config["optimizer"](model.parameters(), **training_config["optimizer_config"])
            scaler = training_config["scaler"](**training_config["scaler_config"])
            best_score = -1e9
            start_epoch = 0
            start_fold = 0

        criterion = training_config["criterion"]()
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,} | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")




        for epoch in range(start_epoch , n_epochs):
            train_loss, train_mse, train_weighted_r2, train_r2s = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, accumulation_steps,
                epoch,n_epochs
            )

           
            val_weighted_r2, val_mse, val_r2s = calculate_weighted_r2(model, val_loader, device)
            for t in CFG.ALL_TARGETS:
                global_writer.add_scalar(f"fold_{fold}/train/r2/{t}", train_r2s.get(t, np.nan), epoch + 1)
                global_writer.add_scalar(f"fold_{fold}/val/r2/{t}", val_r2s.get(t, np.nan), epoch + 1)

            global_writer.add_scalar(f"fold_{fold}/train/weighted_r2", train_weighted_r2, epoch + 1)
            global_writer.add_scalar(f"fold_{fold}/val/weighted_r2", val_weighted_r2, epoch + 1)
            global_writer.add_scalar(f"fold_{fold}/train/mse", train_mse, epoch + 1)
            global_writer.add_scalar(f"fold_{fold}/val/mse", val_mse, epoch + 1)
            global_writer.add_scalar(f"fold_{fold}/train/loss", train_loss, epoch + 1)



            

            improved = False
            if val_weighted_r2 > best_score:
                best_score = val_weighted_r2
                improved = True

            status = "IMPROVED" if improved else "-"
            print(f"[{epoch + 1} / {n_epochs}] |  Train Mse : {train_mse:.4f} | Train R2 : {train_weighted_r2:.4f} | Val Mse : {val_mse:.4f} | Val R2 : {val_weighted_r2:.4f} |  {status} | ",
                  end = "" if ((epoch + 1) % save_every == 0) or (epoch + 1 == n_epochs) else "\n" )

            if ((epoch + 1) % save_every == 0) or (epoch + 1 == n_epochs):
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_score": best_score,
                    "fold": fold,
                    "cfg": {k: v for k, v in CFG.__dict__.items() if not k.startswith("__")},
                    "training_config" : training_config,
                    "fold_seed":fold_seed,
                    "kfold_params":kfold_params
                }
                ckpt_name = f"epoch_{epoch+1:03d}.pth"
                torch.save(ckpt, os.path.join(ckpt_dir, ckpt_name))
                print(f"Saved checkpoint {ckpt_name} for fold {fold+1}")


                
            def _update_df(df , val_r2s, val_weighted_r2 , fold , epoch):
                for target_name , target_r2 in val_r2s.items():
                    df.loc[f"epoch_{epoch + 1}" , [f"fold_{fold+1}_{target_name}"]] = target_r2
                df.loc[f"epoch_{epoch + 1}", [f"total_r2_fold_{fold+1}"]] = val_weighted_r2
                return df.round(2)
            df_table = _update_df(df_table, val_r2s=val_r2s,val_weighted_r2=val_weighted_r2,fold=fold,epoch=epoch)


            if len(df_table)!=0:
                groups = split_df_into_groups(df_table, group_size=n_splits+1)
                for idx, (sub_df , target) in enumerate(groups):
                    tensor_img = df_to_image(sub_df, title=f"target_group {target}")
                    global_writer.add_image(f"target_group {target}", tensor_img, len(groups) + idx, dataformats='CHW')

                global_writer.flush()

            df_table.to_csv(csv_path, index=False)


    df_table.to_csv(csv_path, index=False)


    global_writer.flush()
    global_writer.close()

    print(f"\nSaved per-fold R2 table to: {csv_path}")
    print("K-fold training complete.")
    return { "per_fold_table": df_table}




