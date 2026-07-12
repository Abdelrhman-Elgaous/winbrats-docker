"""
WIN_BRATS v2 inference logic (BraTS-METS 2026) — runs INSIDE the container.
Reads /input, writes flat .nii.gz to /output, weights baked at /opt/weights.
"""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

DATASET_NAME = "Dataset137_BraTS2026MET"
TRAINER_DIR  = "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
FOLDS        = (1, 2, 4)
CHECKPOINT   = "checkpoint_best.pth"

CH  = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}
THR = {"WT": 0.40, "TC": 0.35, "ET": 0.45, "RC": 0.60}
CHANNEL_SUFFIX = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}


def find_trainer_dir(weights_dir: Path) -> Path:
    cand = weights_dir / DATASET_NAME / TRAINER_DIR
    if cand.is_dir():
        return cand
    alt = weights_dir / TRAINER_DIR
    if alt.is_dir():
        return alt
    raise FileNotFoundError(
        f"Could not find '{DATASET_NAME}/{TRAINER_DIR}' under {weights_dir}. "
        f"Contents: {list(weights_dir.rglob('*'))[:20]}"
    )


def load_predictors(trainer_dir: Path):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictors = []
    for f in FOLDS:
        p = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=device,
            verbose=False,
            allow_tqdm=False,
        )
        p.initialize_from_trained_model_folder(
            str(trainer_dir), use_folds=(f,), checkpoint_name=CHECKPOINT
        )
        predictors.append(p)
        print(f"[load] fold {f} predictor ready")
    return predictors


def discover_cases(data_path: Path):
    case_dirs = [d for d in sorted(data_path.iterdir()) if d.is_dir()]
    if not case_dirs:
        case_dirs = [data_path]
    cases = []
    for cdir in case_dirs:
        files = list(cdir.glob("*.nii.gz"))
        found = {}
        for mod in CHANNEL_SUFFIX:
            for fp in files:
                name = fp.name.lower()
                if f"-{mod}." in name or f"_{mod}." in name or name.endswith(f"{mod}.nii.gz"):
                    found[mod] = fp
                    break
        if len(found) != 4:
            print(f"[warn] case {cdir}: found {sorted(found)} (need t1n,t1c,t2w,t2f) -> skipped")
            continue
        cid = (cdir.name if cdir is not data_path
               else found["t1n"].name.split("-t1n")[0].split("_t1n")[0])
        cases.append((cid, found))
    return cases


def predict_case(cid, found, predictors, output_path: Path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for mod, suf in CHANNEL_SUFFIX.items():
            shutil.copy2(found[mod], tmp / f"{cid}_{suf}.nii.gz")
        files = [[str(tmp / f"{cid}_{CHANNEL_SUFFIX[m]}.nii.gz")
                  for m in ("t1n", "t1c", "t2w", "t2f")]]

        probs = []
        for i, p in enumerate(predictors):
            out_dir = tmp / f"out_{i}"
            out_dir.mkdir(exist_ok=True)
            p.predict_from_files(
                files, str(out_dir),
                save_probabilities=True, overwrite=True,
                num_processes_preprocessing=1, num_processes_segmentation_export=1,
            )
            probs.append(np.load(out_dir / f"{cid}.npz")["probabilities"].astype(np.float32))

        ens = np.mean(np.stack(probs, axis=0), axis=0)

        WT = ens[CH["WT"]] >= THR["WT"]
        TC = ens[CH["TC"]] >= THR["TC"]
        ET = ens[CH["ET"]] >= THR["ET"]
        RC = ens[CH["RC"]] >= THR["RC"]
        out = np.zeros(WT.shape, np.uint8)
        out[WT] = 2
        out[TC] = 1
        out[ET] = 3
        out[RC] = 4

        ref = nib.load(str(found["t1n"]))
        out_save = np.transpose(out, (2, 1, 0))
        if out_save.shape != ref.shape:
            raise ValueError(
                f"{cid}: label shape {out_save.shape} != reference {ref.shape} (fail-loud)"
            )
        img = nib.Nifti1Image(out_save, ref.affine, ref.header)
        img.set_data_dtype(np.uint8)
        tmp_out = output_path / f".{cid}.tmp.nii.gz"
        final = output_path / f"{cid}.nii.gz"
        nib.save(img, str(tmp_out))
        os.replace(str(tmp_out), str(final))
        print(f"[ok] {cid} -> {final.name}")


def run_inference(data_path: str, output_path: str, weights: str):
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    weights = Path(weights)

    print("[1/3] locating weights ...")
    trainer_dir = find_trainer_dir(weights)

    print("[2/3] loading 3 persistent predictors (folds 1,2,4) ONCE ...")
    predictors = load_predictors(trainer_dir)

    print("[3/3] discovering + predicting cases ...")
    cases = discover_cases(data_path)
    print(f"      {len(cases)} cases to predict")
    for cid, found in cases:
        predict_case(cid, found, predictors, output_path)

    print("done.")
