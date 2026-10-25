"""
Longitudinal MS new lesion segmentation (FLAIR-only: baseline t0 + follow-up t1)
powered by the 5-fold nnUNetTrainerRobust_500epochs_patience150_v6 model
(5-Fold Hard Vote >=2/5 on checkpoint_best_loss.pth).
"""

import os
import shutil
import tempfile
import nibabel as nib
import numpy as np
import torch

from lst_ai.segment import remove_small_objects


def run_longitudinal_segmentation(
    flair_baseline: str,
    flair_followup: str,
    model_nls_path: str,
    output_segmentation_path: str,
    device: str = "0",
    probmap: bool = False,
    output_prob_path: str = "",
    output_fold_prob_paths=None,
    nls_votes: int = 2,
    lesion_thr: int = 0,
    checkpoint_name: str = "checkpoint_best_loss.pth",
):
    """
    Run 5-fold longitudinal new-lesion segmentation on co-registered, skull-stripped
    baseline (t0) and follow-up (t1) FLAIR images.

    Parameters
    ----------
    flair_baseline : str
        Path to baseline FLAIR image (co-registered & skull-stripped in follow-up space).
    flair_followup : str
        Path to follow-up FLAIR image (skull-stripped in native follow-up space).
    model_nls_path : str
        Directory containing dataset.json, plans.json, and fold_{0..4}/checkpoint_best_loss.pth.
    output_segmentation_path : str
        Output path for binary new-lesion segmentation mask (uint8).
    device : str
        GPU ID (e.g. '0') or 'cpu'.
    probmap : bool
        If True, save ensemble and per-fold probability maps in follow-up FLAIR space.
    output_prob_path : str
        Path to save the ensemble probability map (if probmap=True).
    output_fold_prob_paths : list of str, optional
        Paths to save per-fold probability maps (if probmap=True).
    nls_votes : int
        Minimum number of folds (out of 5) required to vote positive (default: 2 for >=2/5 hard vote).
    lesion_thr : int
        Minimum connected-component lesion size in mm3 (default: 0).
    checkpoint_name : str
        Checkpoint filename inside each fold directory (default: 'checkpoint_best_loss.pth').
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _trainer_mod
    if not hasattr(_trainer_mod, "nnUNetTrainerRobust_500epochs_patience150_v6"):
        setattr(_trainer_mod, "nnUNetTrainerRobust_500epochs_patience150_v6", _trainer_mod.nnUNetTrainer)

    on_cpu = "cpu" in str(device).lower()
    torch_device = torch.device("cpu" if on_cpu else f"cuda:{device}")

    ref_nib = nib.load(flair_followup)
    ref_shape = ref_nib.shape[:3]
    zooms = list(ref_nib.header.get_zooms()[:3])

    tmp_in = tempfile.mkdtemp(prefix="lst_nls_in_")
    tmp_out = tempfile.mkdtemp(prefix="lst_nls_out_")

    try:
        case_id = "sub_nls"
        shutil.copy(flair_baseline, os.path.join(tmp_in, f"{case_id}_0000.nii.gz"))
        shutil.copy(flair_followup, os.path.join(tmp_in, f"{case_id}_0001.nii.gz"))

        votes = np.zeros(ref_shape, dtype=np.uint8)
        prob_accum = np.zeros(ref_shape, dtype=np.float32) if probmap else None

        for f in range(5):
            fold_out_dir = os.path.join(tmp_out, f"fold_{f}")
            os.makedirs(fold_out_dir, exist_ok=True)

            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=not on_cpu,
                device=torch_device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True,
            )
            predictor.initialize_from_trained_model_folder(
                model_nls_path,
                use_folds=(f,),
                checkpoint_name=checkpoint_name,
            )
            predictor.predict_from_files(
                tmp_in,
                fold_out_dir,
                save_probabilities=probmap,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

            fold_seg_path = os.path.join(fold_out_dir, f"{case_id}.nii.gz")
            fold_seg = nib.load(fold_seg_path).get_fdata() > 0
            votes += fold_seg.astype(np.uint8)

            if probmap:
                npz_path = os.path.join(fold_out_dir, f"{case_id}.npz")
                if os.path.exists(npz_path):
                    probs = np.load(npz_path)["probabilities"][1]
                    # nnU-Net stores probabilities in transposed (z, y, x) order
                    if probs.shape != ref_shape and probs.T.shape == ref_shape:
                        probs = probs.T
                    prob_f = probs.astype(np.float32)
                else:
                    prob_f = fold_seg.astype(np.float32)
                prob_accum += prob_f / 5.0
                if output_fold_prob_paths and f < len(output_fold_prob_paths):
                    nib.save(
                        nib.Nifti1Image(prob_f, ref_nib.affine, ref_nib.header),
                        output_fold_prob_paths[f],
                    )

            del predictor
            if not on_cpu:
                torch.cuda.empty_cache()

        seg_bin = (votes >= int(nls_votes)).astype(np.uint8)

        if lesion_thr > 0:
            seg_bin = remove_small_objects(seg_bin, zooms, unit="mm3", thr=lesion_thr).astype(np.uint8)

        out_nii = nib.Nifti1Image(seg_bin, ref_nib.affine, ref_nib.header)
        out_nii.set_data_dtype(np.uint8)
        nib.save(out_nii, output_segmentation_path)

        if probmap and output_prob_path:
            nib.save(
                nib.Nifti1Image(prob_accum, ref_nib.affine, ref_nib.header),
                output_prob_path,
            )

    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)
