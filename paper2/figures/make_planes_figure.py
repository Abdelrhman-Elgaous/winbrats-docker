# ============================================================================
#  Figure for MedIA Statement-of-Significance Q9 (two orthogonal planes):
#  case BraTS-MET-00629-000, whole-tumour reference vs baseline prediction,
#  axial / coronal / sagittal, each panel at the case's own voxel spacing.
#  Reuses the drawing helpers of make_qualitative_figures.py unchanged.
#  Writes  paper2/figures/qualitative/fig_planes_final.pdf / .png
# ============================================================================
import os, sys, shutil, importlib

# --- where make_qualitative_figures.py lives on Drive (edit if different)
SCRIPT_DIR = "/content/drive/MyDrive/brats_2026_metastases/paper2/figures"
sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
sys.path.insert(0, SCRIPT_DIR)

import make_qualitative_figures as mq
importlib.reload(mq)
import native_space as ns

CASE = mq.CASE                      # BraTS-MET-00629-000, same case as Fig. 3
gt, vols, planes, codes, sp = mq.load_case(CASE)
assert len(vols) == 4, "four sequences required"
pred = ns.build_nested_label(ns.load_mean_prob(CASE), suppressor=None)

# standalone figure: no panel letter
mq.panel_letter = lambda ax, letter: None

mq.fig_views(gt, vols, pred, planes, codes, sp)      # writes fig_d_error_views
for ext in ("pdf", "png"):
    src = f"{mq.OUT}/fig_d_error_views.{ext}"
    dst = f"{mq.OUT}/fig_planes_final.{ext}"
    shutil.move(src, dst)
    print("wrote", dst)

print(f"orientation {codes}; spacing (array order) "
      f"{tuple(round(v, 3) for v in sp)}; plane->axis {planes}")
# Copy fig_planes_final.pdf into Overleaf: figures/fig_planes_final.pdf
