"""
nnUNetTrainerSizeAware  ·  EXP-1
================================
Applies the frozen lesion-volume weighting (protocol §7) to the FULL-RESOLUTION
supervised output only.  Deep-supervision losses at coarser scales keep the
original formulation (protocol §8).

THIS DATASET IS REGION-BASED
----------------------------
Dataset137_BraTS2026MET declares four overlapping regions
(WT=[1,2,3], TC=[1,3], ET=[3], RC=[4]), so the supervised target is NOT a
one-channel label map.  nnU-Net converts the label map into one binary channel
per region near the end of the augmentation pipeline.  The target the loss sees
therefore has `num_regions` channels, plus one if an ignore label is used.

That single fact invalidates the obvious implementation of this trainer, and it
is why `_split` below refuses to guess.  See OPEN RISK.

HOW THE WEIGHT MAP TRAVELS
--------------------------
The weight map must receive the SAME spatial augmentation as the label, or it
desynchronises from the lesions it describes.  It therefore rides as a trailing
channel of `seg`, which the augmentation pipeline transforms together with the
label using nearest-neighbour interpolation:

    seg[0]        = the label map        (what the regions are built from)
    seg[1]        = the weight map       (read by the loss, never by the network)

and after region conversion the target is expected to be

    target[0][:, :R]   = the R region channels
    target[0][:, R]    = the weight map

The map is never an input channel and is unavailable at inference.

OPEN RISK — region conversion may discard the weight channel
------------------------------------------------------------
nnU-Net's region transform builds a NEW seg array with exactly `num_regions`
channels and replaces the old one.  In that case the weight channel is silently
dropped, and a naive `_split` would read region channel 1 (the TC mask) as if
it were the weight map — training on WT alone with a binary mask as weights.
It would run to completion and look like a failed hypothesis.

This cannot be settled from the outside: the transform's class, module path and
call signature have all moved between nnU-Net v2 releases.  Settle it against
the installed version before EXP-1:

    python exp1.py inspect-pipeline

`_split` fails loud rather than guessing, so a mis-wired run stops in the first
training step instead of wasting a week.

!!  VERSION-SENSITIVE  !!
The three marked hooks are where nnU-Net v2's trainer API has moved between
releases.  Before spending GPU time:

    python exp1.py selftest

That proves the weighting reaches the loss.  It exercises the loss and the
split, not the augmentation pipeline — `inspect-pipeline` covers that.
"""
import numpy as np
import torch
from torch import nn


#: Weights ride inside `seg`, and nnU-Net's dataloader casts `seg` to int16 on
#: the way out of storage, BEFORE any augmentation:
#:
#:     seg_cropped = torch.from_numpy(
#:         crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)
#:     ).to(torch.int16)
#:
#: (nnunetv2.training.dataloading.data_loader.nnUNetDataLoader, read on
#: 6 October 2026 with `exp1.py inspect-dataloader`.) A float weight of 1.837
#: would arrive as 1, and 1.168 as 1, with nothing raised anywhere: the shapes
#: are unchanged, so `split_target` cannot catch it and training completes.
#:
#: The weights are therefore stored as integers scaled by WEIGHT_SCALE and
#: divided back here. 8.0 * 1000 = 8000 against an int16 ceiling of 32767, and
#: a resolution of 0.001 on a weight in [1, 8] is finer than anything that
#: could matter.
#:
#: The same line pads outside the field of view with -1, which lands in the
#: weight channels too; `SizeAwareDCCE` restores those voxels to the default
#: weight BEFORE dividing.
WEIGHT_SCALE = 1000


# --------------------------------------------------------------- the loss
class SizeAwareDCCE(nn.Module):
    """Soft Dice + BCE, with a per-voxel weight applied to both terms.

    With `weight=None` this must be numerically indistinguishable from the
    stock region-based loss nnU-Net builds for this dataset, which is

        DC_and_BCE_loss({},
                        {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5},
                        dice_class=MemoryEfficientSoftDiceLoss)

    returning `1.0 * bce + 1.0 * dc`, where `dc` is the NEGATIVE mean soft
    Dice.  That equivalence is the whole basis of the controlled comparison:
    if this class and the stock class disagree when unweighted, the two arms
    differ by the formulation as well as by the weighting, and the contrast
    measures neither cleanly.  `exp1.py selftest` now asserts the equivalence
    against the real nnU-Net class whenever nnU-Net is importable.

    AGGREGATION — this is what the first version got wrong
    ------------------------------------------------------
    The configuration sets `batch_dice: false` (protocol §2), so the stock
    loss reduces the soft Dice over the SPATIAL axes only, producing one Dice
    per (sample, region) pair, and then averages those.  Summing globally
    instead — a single Dice over batch, regions and space together — is a
    different objective: it lets a large region in one sample dominate the
    gradient of a small region in another, which is exactly the imbalance this
    experiment exists to manipulate deliberately rather than by accident.  The
    first version summed globally, so the experimental arm would have carried
    two changes relative to the baseline instead of one, and a null result
    would have been uninterpretable.  Recorded as deviation 3 in protocol §15.

    SIGN
    ----
    Stock returns `bce - dice`, not `bce + (1 - dice)`.  The constant does not
    change the gradient, but it changes every number written to the training
    log, and the two arms' loss curves have to be readable against each other.
    So the sign convention is matched exactly.

    THE WEIGHT IS PER REGION
    ------------------------
    `weight` has the same R channels as `logits`, in the same region order, and
    is used channel for channel with no broadcasting. An earlier version
    carried one channel and broadcast it, on the reasoning that the weight
    describes the lesion rather than the region. The coverage audit of
    6 October measured what that costs: components were labelled on the merged
    foreground, so a small enhancing lesion touching non-enhancing tissue sat
    inside a large blob and received weight 1.0 — the background value — for
    2,424 of 6,051 small enhancing-tumour components and 138 of 176 in the
    resection cavity. Weighting each channel by the component the evaluator
    scores in that channel is what removes that mismatch. Protocol §15.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target, weight=None):
        p = torch.sigmoid(logits)
        t = target.float()
        if weight is None:
            w = torch.ones_like(p)
        else:
            w = weight.float()
            # Outside the field of view the dataloader pads with -1, and
            # RemoveLabelTransform maps that to 0. Either value would delete
            # those voxels from the loss in the experimental arm only, which is
            # a difference between the arms that is not the weighting. Restore
            # the default BEFORE dividing, so the guard sees stored units.
            w = torch.where(w > 0, w, torch.full_like(w, float(WEIGHT_SCALE)))
            w = w / WEIGHT_SCALE
            if w.shape != p.shape:
                raise RuntimeError(
                    f"weight has shape {tuple(w.shape)} but the prediction has "
                    f"{tuple(p.shape)}. The weight map must carry one channel "
                    f"per region, in region order. A single broadcast channel "
                    f"is the superseded design (protocol §15); rebuild the maps "
                    f"with `exp1.py weights-pp`.")

        # Spatial axes only: one Dice per (sample, region), then average.
        axes = tuple(range(2, p.ndim))
        inter = (w * p * t).sum(axes)
        denom = (w * p).sum(axes) + (w * t).sum(axes)
        dice = (2.0 * inter + self.smooth) / torch.clip(denom + self.smooth, min=1e-8)
        dice = -dice.mean()

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, t, reduction='none')
        ce = (w * bce).sum() / w.sum().clamp_min(1.0)
        return ce + dice


# ------------------------------------------------------------- the trainer
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
except Exception:                      # import-time safety: selftest needs only the loss
    nnUNetTrainer = object
    DeepSupervisionWrapper = None


# ------------------------------- keeping the weight channels through the
# ------------------------------- region transform
class KeepTrailingSegChannels:
    """Wrap a region transform so trailing `seg` channels survive it.

    WHY THIS EXISTS, measured not assumed
    -------------------------------------
    `exp1.py inspect-pipeline` was run against nnU-Net 2.8.1 on 6 October 2026
    and reported:

        seg channels in : 2  (label + weight)
        seg channels out: 4
        FAIL — the weight channel did NOT survive

    `ConvertSegmentationToRegionsTransform` allocates a NEW array with exactly
    `len(regions)` channels and returns it, so anything riding behind the label
    map is discarded. Without this wrapper the weight map never reaches the
    loss, and EXP-1 would differ from EXP-0 in nothing at all.

    WHY A WRAPPER RATHER THAN A SUBCLASS
    ------------------------------------
    The transform's internals have moved between nnU-Net releases — the class
    has lived in two modules, taken two constructor signatures and two call
    conventions, which is why the probe searches for all of them. A subclass
    would bind to whichever internal shape 2.8.1 happens to have. This wrapper
    depends only on dict-in, dict-out: it hands the inner transform the label
    channel alone, and concatenates the untouched trailing channels back onto
    whatever comes out. Nothing about how the regions are built is reimplemented
    here.

    The kept channels are passed through byte for byte. They are NOT resampled,
    interpolated or converted: the spatial augmentation that must apply to them
    has already happened upstream, where seg is still one array and the weight
    rides with the label under nearest-neighbour interpolation.
    """

    def __init__(self, inner, n_keep: int):
        self.inner = inner
        self.n_keep = int(n_keep)

    @staticmethod
    def _axis(seg):
        # (C, x, y, z) under batchgeneratorsv2, (B, C, x, y, z) under the older
        # pipeline. 3D only, so the two cases are distinguishable by ndim.
        if seg.ndim == 4:
            return 0
        if seg.ndim == 5:
            return 1
        raise RuntimeError(
            f"seg has {seg.ndim} dimensions; this wrapper knows the 3D "
            f"conventions (C,x,y,z) and (B,C,x,y,z) only.")

    def __call__(self, **data_dict):
        key = 'seg' if 'seg' in data_dict else 'segmentation'
        seg = data_dict[key]
        ax = self._axis(seg)
        n_in = seg.shape[ax]
        if n_in <= self.n_keep:
            raise RuntimeError(
                f"seg has {n_in} channels but {self.n_keep} are supposed to be "
                f"trailing weight channels, leaving nothing for the label map. "
                f"The weight map was probably never attached.")

        sl = [slice(None)] * seg.ndim
        sl[ax] = slice(0, n_in - self.n_keep)
        label_only = seg[tuple(sl)]
        sl[ax] = slice(n_in - self.n_keep, n_in)
        kept = seg[tuple(sl)]

        inner_in = dict(data_dict)
        inner_in[key] = label_only
        out = self.inner(**inner_in)
        if not isinstance(out, dict):
            raise RuntimeError(
                f"{type(self.inner).__name__} returned {type(out).__name__}, "
                f"not a dict; this wrapper cannot re-attach the weight "
                f"channels. Re-run `exp1.py inspect-pipeline`.")
        out_key = 'seg' if 'seg' in out else 'segmentation'
        regions = out[out_key]

        if hasattr(regions, 'numpy') or hasattr(kept, 'numpy'):
            import torch as _t
            regions = regions if hasattr(regions, 'numpy') else _t.as_tensor(regions)
            kept = kept if hasattr(kept, 'numpy') else _t.as_tensor(kept)
            kept = kept.to(regions.dtype) if regions.dtype.is_floating_point \
                else kept.float()
            if not regions.dtype.is_floating_point:
                regions = regions.float()
            out[out_key] = _t.cat([regions, kept], dim=ax)
        else:
            out[out_key] = np.concatenate(
                [np.asarray(regions, np.float32), np.asarray(kept, np.float32)],
                axis=ax)
        return out

    def __getattr__(self, name):          # anything else belongs to the inner
        return getattr(self.inner, name)


def patch_region_transform(composed, n_keep: int):
    """Replace the region transform inside a composed pipeline, in place.

    Recomposing the pipeline from scratch would mean reproducing a long
    version-specific argument list; replacing one element of the list that
    nnU-Net already built does not. Refuses unless exactly one region
    transform is found, because zero means the pipeline changed shape and more
    than one means the weight channels would be concatenated twice.
    """
    seen = []

    def walk(node):
        lst = getattr(node, 'transforms', None)
        if lst is None:
            return
        for i, t in enumerate(lst):
            if type(t).__name__ == 'ConvertSegmentationToRegionsTransform':
                lst[i] = KeepTrailingSegChannels(t, n_keep)
                seen.append(t)
            else:
                walk(t)

    walk(composed)
    if len(seen) != 1:
        raise RuntimeError(
            f"found {len(seen)} ConvertSegmentationToRegionsTransform in the "
            f"composed pipeline; expected exactly 1. The augmentation pipeline "
            f"is not the shape this trainer was written against. Run "
            f"`exp1.py inspect-pipeline` and do not train until this is "
            f"resolved.")
    return composed


def split_target(target, n_out: int, expect_weight: bool = True):
    """Separate the R supervised channels from the R trailing weight channels.

    `n_out` is the number of channels the network predicts — for this dataset
    the number of regions. The weight map carries ONE CHANNEL PER REGION
    (protocol §15, pre-experimental coverage audit): a component is labelled
    inside each scored region separately, as the evaluator labels it, so the
    weight a voxel receives in the enhancing-tumour channel is derived from the
    enhancing-tumour component it belongs to and not from the merged
    foreground blob that contains it.

    The target therefore has `2 * n_out` channels: R region masks followed by R
    weight maps, in the same region order. Deciding by channel count alone is
    what makes a region-based run go wrong silently, so the expected width is
    passed in and anything else is an error.

    Returns (label, weight, lower_resolution_labels).
    """
    if isinstance(target, (list, tuple)):
        full, rest = target[0], list(target[1:])
    else:
        full, rest = target, []

    got = full.shape[1]
    if not expect_weight:
        return full[:, :n_out], None, [t[:, :n_out] for t in rest]

    if got == 2 * n_out:
        return (full[:, :n_out], full[:, n_out:2 * n_out],
                [t[:, :n_out] for t in rest])

    if got == n_out:
        raise RuntimeError(
            f"the supervised target has {got} channels, exactly the number the "
            f"network predicts, so NO weight channel survived the augmentation "
            f"pipeline. The region transform rebuilds `seg` and discards "
            f"trailing channels — measured, not assumed, on 6 October 2026. "
            f"`nnUNetTrainerSizeAware` installs `KeepTrailingSegChannels` to "
            f"prevent exactly this, so seeing it here means the patch did not "
            f"take. Run `python exp1.py inspect-pipeline` and fix it before "
            f"training; do not fall back to the unweighted loss, because that "
            f"silently turns EXP-1 into EXP-0.")

    if got == n_out + 1:
        raise RuntimeError(
            f"the supervised target has {n_out} region channels plus 1 weight "
            f"channel. That is the SUPERSEDED single-channel weight map, built "
            f"from connected components of the merged foreground. The coverage "
            f"audit of 6 October measured that definition reaching only 59.9% "
            f"of small enhancing-tumour components and 21.6% in the resection "
            f"cavity. Rebuild the maps with `exp1.py weights-pp`, which now "
            f"writes one channel per region.")

    raise RuntimeError(
        f"the supervised target has {got} channels; expected {n_out} region "
        f"channels plus {n_out} weight channels = {2 * n_out}. Refusing to "
        f"guess which channels are weights: guessing wrong trains on one "
        f"region with a binary mask as weights and still completes. Run "
        f"`python exp1.py inspect-pipeline`.")


class SizeAwareDeepSupervisionLoss(nn.Module):
    """The object that `self.loss` must be.

    DISCOVERED AT EPOCH 0 OF THE EXPERIMENTAL ARM, 11 OCTOBER 2026.

    Until now the size-aware objective lived in
    `nnUNetTrainerSizeAware.compute_loss`, on the assumption that nnU-Net's
    training step calls a `compute_loss` hook. It does not. nnU-Net 2.8.1's
    `nnUNetTrainer.train_step` (line 1037) calls

        l = self.loss(output, target)

    inline, and `validation_step` does the same. `compute_loss` was therefore
    never reached by anything except `exp1.py dry-run`, which called it
    directly -- so pre-flight gate 6 passed on a code path the training loop
    bypasses. In the real run the stock loss received the 8-channel
    label+weight target and died inside a scatter bounded by the 4 regions:

        Assertion `index out of bounds: 0 <= tmp1 < 4` failed
        triton_poi_fused__to_copy_scatter_zeros_1 ... 110100480
        110100480 = 2 batch x 8 channels x 160x224x192

    The crash is the lucky outcome. Had the stock loss tolerated the extra
    channels, the arm would have trained to completion with the weight map
    ignored, and EXP-1 would have differed from the baseline in nothing at
    all -- the precise failure the `split_target` error messages were written
    to prevent, arriving by a route none of them could see.

    So the objective is installed where nnU-Net actually looks for it.
    Everything else in `train_step` -- autocast, the gradient scaler, gradient
    clipping, the optimizer step -- stays stock, and therefore stays identical
    to the baseline arm, which is what Section 3 requires.
    """

    def __init__(self, base, size_aware, n_out: int):
        super().__init__()
        self.base = base                  # stock loss, DS-wrapped, .dc compiled
        self.size_aware = size_aware
        self.n_out = int(n_out)
        # Declared by train_step / validation_step, never inferred from the
        # channel count. Inferring is what would let a pipeline that silently
        # drops the weight channels fall back to the stock loss and train an
        # arm identical to the baseline without saying so. The phase is stated
        # and the width has to match it.
        self.expect_weights = True

    def forward(self, output, target):
        if not self.expect_weights:
            # Validation. The targets carry no weight channels and the
            # objective is the stock one, identical in both arms -- see HOOK 4
            # for why validation is deliberately left unpatched.
            #
            # The width is asserted rather than assumed. If the validation
            # transforms were ever patched again, the stock loss would receive
            # a 2R-channel target and scatter out of bounds the way the
            # training loop did at epoch 0 on 11 October 2026 -- and the
            # traceback would point at nnU-Net's own loss, not at this file.
            full = target[0] if isinstance(target, (list, tuple)) else target
            got = full.shape[1]
            if got != self.n_out:
                raise RuntimeError(
                    f"validation target has {got} channels, expected "
                    f"{self.n_out}. Validation is deliberately NOT patched "
                    f"(deviation 11): the weight channels must not reach it, "
                    f"because nnUNetTrainer.validation_step also feeds the "
                    f"target to get_tp_fp_fn_tn against an {self.n_out}-"
                    f"channel prediction. If this says {2 * self.n_out}, "
                    f"`get_validation_transforms` has been patched again — "
                    f"remove the patch rather than widening this check.")
            return self.base(output, target)

        label, weight, lower = split_target(target, self.n_out)
        if weight is None:
            raise RuntimeError(
                "split_target returned no weight map while weights were "
                "expected. Do not continue: the unweighted loss turns EXP-1 "
                "into EXP-0.")

        out_full = output[0] if isinstance(output, (list, tuple)) else output
        l_full = self.size_aware(out_full, label, weight)

        if not (isinstance(output, (list, tuple)) and len(output) > 1 and lower):
            return l_full

        # nnU-Net normalises its deep-supervision weights to sum to 1 and sets
        # the coarsest scale to 0. Re-deriving or re-normalising them changes
        # the loss scale, which changes the effective learning rate -- a second
        # difference between the arms, which Section 3 forbids. Take them as
        # they are.
        ds_w = getattr(self.base, 'weight_factors', None)
        if ds_w is None:
            raise RuntimeError(
                "the deep-supervision wrapper exposes no `weight_factors`; "
                "reconstructing them would change the loss scale relative to "
                "the baseline arm and break the controlled comparison (§3). "
                "Locate the attribute in your nnU-Net version and set it here.")
        ds_w = [float(x) for x in ds_w]

        # The coarser scales must be BIT-IDENTICAL to the baseline arm, because
        # §8 says they keep the original formulation. Calling `self.size_aware`
        # with weight=None would be numerically equal now that the aggregation
        # is fixed, but "numerically equal" is a claim to be avoided when the
        # stock callable is right here: `DeepSupervisionWrapper` holds it as
        # `.loss`. Use it, and the coarse scales stop being a place where the
        # arms can drift apart at all.
        stock = getattr(self.base, 'loss', None)
        if stock is None:
            raise RuntimeError(
                "the deep-supervision wrapper exposes no `.loss`; the coarse "
                "scales cannot be made identical to the baseline arm without "
                "it (§8). Locate the attribute in your nnU-Net version.")

        total = ds_w[0] * l_full
        for i, (o, t) in enumerate(zip(output[1:], lower), start=1):
            if ds_w[i] == 0.0:                 # nnU-Net zeroes the coarsest
                continue
            total = total + ds_w[i] * stock(o, t)
        return total


class nnUNetTrainerCkpt10(nnUNetTrainer):
    """The stock trainer, checkpointing every 10 epochs instead of every 50.

    THIS IS THE BASELINE ARM.  It changes nothing that touches the model: not
    the loss, the data, the augmentation, the optimizer or the schedule.  It
    changes only how often `checkpoint_latest.pth` is written, which bounds how
    much work a Colab session drop can destroy -- at 274 s an epoch, 50 epochs
    is 3.8 hours of exposure and 10 epochs is 46 minutes.

    `nnUNetTrainerSizeAware` inherits from this class rather than from
    `nnUNetTrainer`, so both arms share the checkpoint frequency and protocol
    §3's requirement that the arms differ only in the loss still holds.  If you
    run the baseline with the stock `nnUNetTrainer` and the experimental arm
    with `nnUNetTrainerSizeAware`, that requirement is broken -- not because
    checkpoint frequency affects training, but because the arms would then also
    differ in the class they were run from, and there would be no record of
    which difference mattered.

    Recorded as deviation 2 in protocol §15.

    WHY `__init__` IS NOT OVERRIDDEN
    nnUNetTrainer.__init__ records its own arguments by introspecting
    `inspect.signature(self.__init__)` and then reading those names out of
    `locals()`.  A subclass whose `__init__` is `(*args, **kwargs)` makes that
    introspection return `args` and `kwargs`, which are not locals of the base
    `__init__`, and construction dies with `KeyError: 'args'` before training
    starts.  Restating the base signature instead would couple this file to a
    signature that has changed between nnU-Net releases.

    `save_every` is therefore set in `on_train_start`, which runs once before
    the epoch loop and again after `--c` resumes, and is read by `on_epoch_end`
    afterwards.  The change is written to the training log so the run carries
    its own evidence.
    """
    def on_train_start(self):
        super().on_train_start()
        self.save_every = 10
        self.print_to_log_file(
            f"save_every = {self.save_every} (protocol section 15, deviation 2)")


class nnUNetTrainerSizeAware(nnUNetTrainerCkpt10):
    """The baseline arm plus the weighted full-resolution loss term.

    Inherits the checkpoint frequency so that the only intended difference
    between the arms remains the loss (protocol §3).
    """

    # ---- HOOK 1 ---------------------------------------------------------
    # How wide the supervised target is.  Confirm the attribute name against
    # your nnU-Net version; `num_segmentation_heads` has been stable, but the
    # label_manager API has moved.
    def _n_out(self):
        lm = self.label_manager
        for attr in ('num_segmentation_heads',):
            if hasattr(lm, attr):
                return int(getattr(lm, attr))
        if getattr(lm, 'has_regions', False):
            return len(lm.foreground_regions)
        raise RuntimeError("cannot determine the number of supervised output "
                           "channels from this label_manager; set it explicitly")

    # ---- HOOK 2 ---------------------------------------------------------
    def _build_loss(self):
        # `self.loss` is what nnUNetTrainer.train_step calls, so the
        # size-aware objective has to BE this object. Returning the stock loss
        # here and putting the objective in `compute_loss` is what made the
        # weights unreachable until 11 October 2026 (deviation 11).
        base = super()._build_loss()           # stock DS-wrapped loss
        self._size_aware = SizeAwareDCCE()
        self._base_loss = base
        return SizeAwareDeepSupervisionLoss(base, self._size_aware,
                                            self._n_out())

    # ---- HOOK 3 ---------------------------------------------------------
    # nnU-Net has NO `compute_loss` hook: train_step and validation_step both
    # call `self.loss(output, target)` inline. This method is kept only so
    # that `exp1.py dry-run` and the training loop exercise the SAME object,
    # which is the whole reason gate 6 missed deviation 11 — it tested a
    # method nothing else called. Do not put logic here.
    def compute_loss(self, output, target):
        return self.loss(output, target)

    # ---- HOOK 4 ---------------------------------------------------------
    # Keep the weight channels alive through the region transform, IN TRAINING
    # ONLY.
    #
    # `inspect-pipeline` measured on 6 October 2026 that
    # ConvertSegmentationToRegionsTransform rebuilds `seg` with exactly
    # len(regions) channels and drops everything behind the label map. Without
    # this, the weight never reaches the loss. The stock pipeline is built
    # first and then one element of it is replaced, rather than recomposed
    # here, because recomposing means reproducing a long version-specific
    # argument list that has changed between nnU-Net releases.
    #
    # VALIDATION IS NO LONGER PATCHED (deviation 11, 11 October 2026). The
    # earlier note here read: "Validation gets the same patch. The validation
    # loss is logged and compared between arms, so it has to be the same
    # objective." The conclusion contradicted its own premise. If the
    # experimental arm's val_loss is the size-aware objective and the
    # baseline's is the stock one, the two logged curves are NOT comparable;
    # leaving validation stock makes them measure the same thing in both arms.
    # The intervention is a training-time reweighting, and validation is
    # measurement -- both arms should be measured with the same yardstick.
    #
    # It also removes a defect. nnUNetTrainer.validation_step uses `target`
    # twice: once for the loss and once for the online pseudo-dice, where
    # `get_tp_fp_fn_tn` multiplies a 4-channel prediction by it. An 8-channel
    # target fails there regardless of what the loss does.
    def get_training_transforms(self, *a, **k):
        return patch_region_transform(
            super().get_training_transforms(*a, **k), self._n_out())

    # ---- HOOK 5 ---------------------------------------------------------
    # Which phase the loss is in, declared rather than inferred. Both are thin
    # delegations on purpose: every line of nnU-Net's own train_step and
    # validation_step -- autocast, the gradient scaler, clipping, the
    # optimizer step, the pseudo-dice bookkeeping -- stays stock, and so stays
    # identical to the baseline arm (§3). Copying those bodies to insert one
    # call is how the arms would quietly come to differ in a second place.
    def train_step(self, batch):
        if not hasattr(nnUNetTrainer, 'train_step'):
            raise RuntimeError("nnU-Net not importable; this class needs it")
        self.loss.expect_weights = True
        return super().train_step(batch)

    def validation_step(self, batch):
        self.loss.expect_weights = False
        return super().validation_step(batch)


# ------------------------------------------- preprocessing-time injection
def attach_weight_channel(*_a, **_k):
    """Retired 6 October 2026. Use `exp1.py weights-pp`.

    This wrote a single float weight channel built from components of the
    merged foreground. Both halves of that are now wrong:

    - the map must carry ONE CHANNEL PER REGION, built from components inside
      each scored region (protocol §15, deviation 7). The merged definition
      reached only 59.9% of small enhancing-tumour components;
    - the weights must be stored as integers scaled by WEIGHT_SCALE. nnU-Net's
      dataloader casts `seg` to int16 before any augmentation, so a float
      weight of 1.837 arrives at the loss as 1, silently.

    Left as a loud failure rather than deleted, because a function that writes
    a plausible-looking weight map the wrong way is worse than no function.
    """
    raise RuntimeError(
        "attach_weight_channel is retired: it wrote one float weight channel "
        "from merged-foreground components. Use `exp1.py weights-pp`, which "
        "writes one int16 channel per region scaled by WEIGHT_SCALE. See "
        "protocol §15, deviation 7.")


def _attach_weight_channel_superseded(seg_npy_path: str, weight_npy_path: str) -> None:
    """Append the weight map as a trailing seg channel, in place.

    Prefer `exp1.py weights-pp`, which builds the map on the preprocessed grid
    so no resampling step can misalign it.  This path exists only for maps you
    have already resampled yourself.

    The array is promoted to float32: preprocessed seg is an integer type, and
    casting weights into it would truncate every weight below 1 to 0 and every
    weight above 1 to its floor — the intervention would quietly disappear.
    """
    seg = np.load(seg_npy_path)
    w = np.load(weight_npy_path)
    if seg.ndim == 3:
        seg = seg[None]
    if seg.shape[0] >= 2:
        return                                  # already attached
    if w.shape != seg.shape[1:]:
        raise ValueError(f"weight map {w.shape} does not match seg {seg.shape[1:]}; "
                         "resample it onto the preprocessed grid first")
    np.save(seg_npy_path,
            np.concatenate([seg.astype(np.float32), w[None].astype(np.float32)], 0))
