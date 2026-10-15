#!/usr/bin/env python3
"""Migrate active gallery/config references; retain historical model taxonomy."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pickletools
import shutil
import sys
import zipfile

import joblib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from migrate_macaque_identities import AUDIT, DATA, REPO, REGISTRY, SETS, read, write, sha
from src.datasets.identity_registry import IdentityRegistry


def finalize():
    if not (AUDIT / "published.json").is_file():
        raise ValueError("Publish canonical datasets first")
    if (AUDIT / "reference_verification.json").exists():
        raise FileExistsError("Reference migration already complete")
    registry = IdentityRegistry(REGISTRY)
    plan = read(AUDIT / "plan.json")
    before = AUDIT / "before"
    artifacts = REPO / "animal-face-id/artifacts"
    saved_artifacts = before / "artifacts"
    saved_artifacts.mkdir(exist_ok=True)
    original_names = read(before / "classifier_label_order_156.json")["class_names"]
    assert len(original_names) == 156
    # Inspect only checkpoint metadata, without allocating model weights or GPUs.
    checkpoints = {}
    for path in artifacts.glob("macaque*.pt"):
        with zipfile.ZipFile(path) as archive:
            data = archive.read(next(n for n in archive.namelist() if n.endswith("data.pkl")))
        ops = list(pickletools.genops(data))
        counts = []
        for i, (op, arg, position) in enumerate(ops):
            if arg == "num_classes":
                counts.append(next(value for operation, value, _ in ops[i+1:i+6]
                                   if operation.name in ("BININT1", "BININT2", "BININT", "INT")))
        if counts != [156]:
            raise ValueError(f"Review classifier order for {path}: {counts}")
        checkpoints[path.name] = sha(path)
    # Gallery entries must reference surviving, reviewed training crops. Rows for
    # removed/missing photos are excluded, rather than given an invented path.
    prefix = artifacts / "index/macaque_auto"
    metadata_file = Path(str(prefix) + "_meta.json")
    embeddings_file = Path(str(prefix) + "_embeddings.npy")
    metadata = read(metadata_file)
    embeddings = np.load(embeddings_file)
    npz_file = artifacts / "train_embeddings.npz"
    npz = np.load(npz_file, allow_pickle=False)
    assert np.array_equal(npz["embeddings"], embeddings)
    assert list(npz["labels"]) == metadata["labels"]
    galleries = {p: joblib.load(p) for p in artifacts.glob("*gallery*.pkl")}
    gallery_provenance = {}
    gallery_original_arrays = {}
    for path, gallery in galleries.items():
        assert gallery["gallery_ids"] == metadata["labels"], path
        gallery_provenance[path] = np.array_equal(gallery["model"]._fit_X, embeddings)
        gallery_original_arrays[path] = gallery["model"]._fit_X.copy()
    crops = plan["datasets"]["reviewed_crops"]
    canonical_records = {r["path"]: r for rs in crops["after"].values() for r in rs}
    keep, labels, meta, omitted = [], [], [], []
    for index, (label, item) in enumerate(zip(metadata["labels"], metadata["meta"])):
        path = item.get("image_path", "")
        marker = "/macaque_crops/"
        if marker not in path:
            raise ValueError(f"Unrecognized gallery provenance: {path}")
        relative = path.split(marker, 1)[1]
        target = crops["path_map"].get(relative)
        if target is None:
            omitted.append({"original_row": index, "label": label, "image_path": path,
                            "reason": "Original gallery crop no longer belongs to the active reviewed dataset"})
            continue
        record = canonical_records[target]
        assert Path(target).parts[0] == "train"
        assert registry.resolve(label) == record["id"]
        keep.append(index)
        labels.append(record["id"])
        meta.append({**item, **registry.metadata(record["id"]),
                     "image_path": str(SETS["reviewed_crops"][0] / target),
                     "source_path": record["source_path"],
                     "master_source_path": record["master_source_path"],
                     "original_gallery_row": index})
    assert set(labels) == set(registry.identities)
    retained = embeddings[keep]
    for path in [metadata_file, embeddings_file, npz_file, *galleries]:
        destination = saved_artifacts / path.relative_to(artifacts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    np.save(embeddings_file, retained)
    write(metadata_file, {"labels": labels, "meta": meta,
          "identity_registry_version": registry.payload["version"],
          "embedding_policy": "Historical embeddings retained exactly; obsolete crop references excluded"})
    np.savez(npz_file, embeddings=retained, labels=np.array(labels), paths=np.array([m["image_path"] for m in meta]))
    for path, gallery in galleries.items():
        if gallery_provenance[path]:
            gallery["model"].fit(retained)
            gallery["gallery_ids"] = labels
            gallery["image_paths"] = [m["image_path"] for m in meta]
        else:
            # A different backbone produced these embeddings. Their filenames
            # were never saved: only canonicalize IDs; do not invent row paths
            # or replace features with embeddings from another model.
            gallery["gallery_ids"] = [registry.resolve(n) for n in gallery["gallery_ids"]]
            gallery["image_provenance"] = "Historical gallery lacks image references; names canonicalized, own embeddings preserved"
        gallery["identity_registry_version"] = registry.payload["version"]
        joblib.dump(gallery, path)
    write(AUDIT / "gallery_omitted_references.json", omitted)
    # Frozen configs are for historical evaluation; active configs train 149
    # classes with fresh names so they cannot overwrite 156-output checkpoints.
    for path in (REPO / "animal-face-id/configs").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        if config["data"].get("dataset_name") != "macaque_faces":
            continue
        saved = before / "configs" / path.name
        frozen = yaml.safe_load(saved.read_text())
        frozen["data"]["raw_root"] = str(before / "reviewed_crops")
        frozen["data"]["splits_path"] = str(before / "reviewed_crops_splits.json")
        frozen["trainer"]["checkpoint_dir"] = str(before / "historical_config_training_outputs")
        saved.write_text("# Historical 156-identity dataset; use for evaluation of previous runs.\n" + yaml.safe_dump(frozen, sort_keys=False))
        config["name"] += "_canonical_20261014"
        config["data"]["num_classes"] = 149
        config["data"]["identity_registry"] = str(REGISTRY)
        config["inference"]["checkpoint"] = "artifacts/" + config["name"] + "_best.pt"
        # Existing GUI/CLI can explicitly load any old checkpoint; inference sums
        # aliases using its bound original label order, never renumbers its head.
        config["inference"]["gallery_path"] = "artifacts/" + config["name"] + "_gallery.pkl"
        path.write_text("# Canonical names/current groups from the updated master; existing split membership retained.\n" + yaml.safe_dump(config, sort_keys=False))
    for name, digest in checkpoints.items():
        write(artifacts / (name + ".labels.json"), {"checkpoint_sha256": digest,
            "class_names": original_names, "taxonomy": "Original 156 labels before 2026-10-14 canonical migration",
            "canonical_registry": str(REGISTRY)})
    # Validate every active gallery reference and embedding after persistence.
    loaded = read(metadata_file)
    assert loaded["labels"] == labels
    assert np.array_equal(np.load(embeddings_file), embeddings[keep])
    assert all(Path(m["image_path"]).is_file() for m in loaded["meta"])
    for path in galleries:
        payload = joblib.load(path)
        expected = labels if gallery_provenance[path] else [registry.resolve(n) for n in metadata["labels"]]
        assert payload["gallery_ids"] == expected
        features = embeddings[keep] if gallery_provenance[path] else gallery_original_arrays[path]
        assert np.array_equal(payload["model"]._fit_X, features)
    for path in artifacts.glob("macaque*.pt"):
        assert sha(path) == checkpoints[path.name]
    report = {"canonical_identities": 149, "old_classifier_outputs": 156,
              "checkpoints_preserved_and_bound_to_original_label_order": len(checkpoints),
              "gallery_original_rows": len(embeddings), "gallery_current_rows": len(keep),
              "gallery_omitted_obsolete_references": len(omitted), "gallery_embeddings_changed": 0,
              "all_gallery_references_exist": True, "active_training_classes": 149,
              "historical_metrics_preserved": True}
    report["historical_galleries_without_image_provenance"] = [str(p) for p, known in gallery_provenance.items() if not known]
    write(AUDIT / "reference_verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    finalize()
