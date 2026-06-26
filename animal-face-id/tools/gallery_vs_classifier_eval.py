"""Reconcile the closed-set classifier metric with the GUI's gallery-kNN metric.

Method A (closed-set classifier): argmax cosine(test_emb, ArcFace prototype weights).
   - Independent numpy recompute of the corrected eval => sanity-checks the 0.83 fix.
Method B (gallery k-NN, the GUI's method): nearest train-image embedding by cosine.
   - With and without an open-set similarity threshold (GUI rejects low-similarity).

Reuses artifacts/train_embeddings.npz as the gallery if it matches the checkpoint.
"""
from __future__ import annotations
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config.base import load_config
from src.datasets.dataset_registry import build_dataloader
from src.models.backbones import build_backbone

CKPT = "artifacts/macaque-resnet50-arcface_aug2_best.pt"
CFG = "configs/train_macaque_arcface_aug2.yaml"
DEVICE = "cpu"

def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def main():
    cfg = load_config(CFG).as_dict()
    dev = torch.device(DEVICE)
    ckpt = torch.load(CKPT, map_location=dev)
    proto = ckpt["head_state"]["weight"].cpu().numpy()  # (156,256) ArcFace prototypes
    print("prototypes:", proto.shape)

    # backbone
    model = build_backbone(cfg["model"]["backbone"], cfg["model"]["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True); model.to(dev); model.eval()

    # gallery from precomputed train embeddings (verify it matches this checkpoint)
    g = np.load("artifacts/train_embeddings.npz", allow_pickle=True)
    gemb, glab = g["embeddings"], np.array([str(x) for x in g["labels"]])
    # class order = sorted unique ids (matches dataset.class_to_idx)
    classes = sorted(set(glab))
    cidx = {c: i for i, c in enumerate(classes)}
    # sanity: closed-set train top1 using prototypes (should be high if same ckpt)
    sim_train = l2n(gemb) @ l2n(proto).T
    train_top1 = (sim_train.argmax(1) == np.array([cidx[l] for l in glab])).mean()
    print(f"[check] gallery train closed-set top1 = {train_top1:.3f} (high => npz matches checkpoint)")

    # embed test fresh
    tl = build_dataloader("macaque_faces", "test", {**cfg["data"], "batch_size": 64, "num_workers": 8},
                          shuffle=False, drop_last=False)
    temb, tlab = [], []
    with torch.no_grad():
        for b in tl:
            temb.append(model(b["image"].to(dev)).cpu().numpy()); tlab += list(b["id"])
    temb = np.concatenate(temb); tlab = np.array(tlab)
    y = np.array([cidx[l] for l in tlab])
    print("test embeddings:", temb.shape)

    # Method A: closed-set classifier (prototypes)
    simA = l2n(temb) @ l2n(proto).T
    topA = (simA.argmax(1) == y).mean()
    print(f"\nMethod A  closed-set classifier (prototypes): top1 = {topA:.3f}")

    # Method B: gallery kNN (k=1) cosine vs train images
    Gn = l2n(gemb); Tn = l2n(temb)
    sims = Tn @ Gn.T                      # (n_test, n_gallery) cosine
    nn = sims.argmax(1)
    predB = glab[nn]; simB = sims.max(1)
    topB = (predB == tlab).mean()
    print(f"Method B  gallery kNN (k=1, no threshold):    top1 = {topB:.3f}")

    # Method B + open-set threshold (GUI rejects below thr as 'unknown')
    for thr in (0.5, 0.6, 0.7, 0.75):
        acc_all = ((predB == tlab) & (simB >= thr)).mean()   # rejected counts as wrong
        kept = (simB >= thr).mean()
        acc_kept = (predB[simB >= thr] == tlab[simB >= thr]).mean() if kept > 0 else float("nan")
        print(f"Method B + thr={thr:.2f}: kept={kept*100:4.1f}%  acc(all)={acc_all:.3f}  acc(kept)={acc_kept:.3f}")

    # kNN majority vote k=5 (closer to GUI top-k)
    k = 5
    idxk = np.argsort(-sims, axis=1)[:, :k]
    from collections import Counter
    predk = np.array([Counter(glab[row]).most_common(1)[0][0] for row in idxk])
    print(f"Method B  gallery kNN (k=5 majority):         top1 = {(predk==tlab).mean():.3f}")

if __name__ == "__main__":
    main()
