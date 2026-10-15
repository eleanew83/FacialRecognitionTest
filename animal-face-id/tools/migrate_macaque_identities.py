#!/usr/bin/env python3
"""Audit, then publish canonical identities without changing images or splits.

The old datasets are moved into the audit's before/ directory. New generations
use hard links, so image bytes and mtimes remain unchanged. Never edit those
linked image files in place; generate replacement datasets in fresh directories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "animal-face-id"))
from src.datasets.identity_registry import IdentityRegistry, unique_destination
from src.datasets.split_integrity import file_fingerprint, source_photo_key, validate_manifest
from src.datasets.detection_integrity import parse_labels

DATA = REPO / "animal-face-id/data/macaque_faces"
AUDIT = DATA / "identity_migration_2026_10_14"
OLD_AUDIT = DATA / "source_split_repair_2026_10_07"
MASTER_AUDIT = DATA / "source_collection_cleanup_2026_10_13"
MASTER = REPO.parent / "Gibraltar_Macaques_Photos_Cleaned_Updated"
REGISTRY = DATA / "identity_registry.json"
SETS = {
    "source": (REPO / "macaque_split_data", REPO / "macaque_split_data/splits.json"),
    "reviewed_crops": (REPO / "yolo_detection/yolo_detection_code/output/macaque_crops", DATA / "splits.json"),
    "detector_20261012_crops": (
        REPO / "yolo_detection/yolo_detection_code/output/macaque_crops_macaque_face_detector_20261012_job37715504",
        REPO / "yolo_detection/yolo_detection_code/output/macaque_crops_macaque_face_detector_20261012_job37715504/splits.json"),
}
DETECTOR = REPO / "yolo_detection/yolo_detection_data"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".migration-tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def canonical_registry():
    identities = {}
    for group in sorted(MASTER.iterdir()):
        if not group.is_dir() or group.name.startswith("."):
            continue
        for gender in sorted(group.iterdir()):
            if not gender.is_dir() or gender.name.startswith("."):
                continue
            for individual in sorted(gender.iterdir()):
                if not individual.is_dir() or individual.name.startswith("."):
                    continue
                if individual.name in identities:
                    raise ValueError(f"Individual appears in multiple master folders: {individual.name}")
                identities[individual.name] = {
                    "current_group": group.name, "gender": gender.name,
                    "source_directory": str(individual.relative_to(MASTER)),
                }
    aliases = {r["existing_identity"]: r["current_source_identity"]
               for r in rows(MASTER_AUDIT / "identity_label_crosswalk.csv")}
    assert len(identities) == 149 and len(aliases) == 156
    assert set(aliases.values()) == set(identities)
    return {"version": "2026-10-14", "source_root": str(MASTER),
            "group_policy": "Current group comes from the master folder; filenames retain capture history.",
            "identities": identities, "aliases": aliases}


def inventory(root):
    result = set()
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if not name.startswith(".") and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                result.add(str((Path(folder) / name).relative_to(root)))
    return result


def checked_bytes(item):
    path, expected = item
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if sig(before) != sig(after) or digest != expected["sha256"]:
        raise ValueError(f"Image changed since the previous audit: {path}")
    if (after.st_size, after.st_mtime_ns) != (expected["file_bytes"], expected["mtime_ns"]):
        raise ValueError(f"Image stat changed: {path}")
    return digest


def make_plan():
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / ".gitignore").write_text("*\n")
    if (AUDIT / "verification.json").exists():
        raise FileExistsError("Migration is already complete; use --verify")
    write(REGISTRY, canonical_registry())
    registry = IdentityRegistry(REGISTRY)
    master_rows = {r["existing_source_path"]: r for r in rows(MASTER_AUDIT / "source_to_existing_splits.csv")}
    original_fingerprints = {r["path"]: r for r in jsonl(OLD_AUDIT / "fingerprints.jsonl")}
    repair = read(OLD_AUDIT / "repair_plan.json")
    source_details = {entry["target"]: entry["record"] for entry in repair["retained"]}
    crop_sources = defaultdict(set)
    reviewed_sources = defaultdict(set)
    new_crop_sources = defaultdict(set)
    sha_sources, pixel_sources = defaultdict(set), defaultdict(set)
    for retained in repair["retained"]:
        target = retained["target"]
        canonical = registry.resolve(Path(target).parts[1])
        for old_path in retained["group_original_paths"]:
            original = original_fingerprints[old_path]
            sha_sources[original["sha256"]].add(target)
            pixel_sources[original["pixels_sha256"]].add(target)
            for crop in original["current_crop_reference"]:
                crop_sources[(canonical, source_photo_key(crop["path"]))].add(target)
                reviewed_sources[crop["path"]].add(target)
    source_before = read(SETS["source"][1])
    for records in source_before.values():
        for record in records:
            canonical = registry.resolve(record["id"])
            crop_sources[(canonical, source_photo_key(record["path"]))].add(record["path"])
            source_name = Path(record["path"])
            crop_name = source_name.with_name(source_name.stem + "_crop0" + source_name.suffix)
            new_crop_sources[str(crop_name)].add(record["path"])
            sha_sources[record["sha256"]].add(record["path"])
            pixel_sources[record["pixels_sha256"]].add(record["path"])
    assert set(master_rows) == {r["path"] for rs in source_before.values() for r in rs}
    plan = {"registry": registry.payload, "datasets": {}, "detector": [], "unresolved": []}
    path_maps, checked = {}, 0
    for name, (root, manifest_path) in SETS.items():
        before = read(manifest_path)
        validate_manifest(before)
        assert inventory(root) == {r["path"] for rs in before.values() for r in rs}, name
        after = {s: [] for s in ("train", "val", "test")}
        occupied, mapping = set(), {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            checks = [(root / r["path"], r) for rs in before.values() for r in rs]
            for _ in pool.map(checked_bytes, checks):
                checked += 1
                if checked % 1000 == 0:
                    print(f"Verified image bytes: {checked}", flush=True)
        for split, records in before.items():
            for record in records:
                canonical = registry.resolve(record["id"])
                if name == "source":
                    source_path = record["path"]
                else:
                    candidates = (reviewed_sources if name == "reviewed_crops" else new_crop_sources)[record["path"]]
                    if len(candidates) != 1:
                        plan["unresolved"].append({"dataset": name, "path": record["path"], "source_candidates": sorted(candidates)})
                        continue
                    source_path = next(iter(candidates))
                master = master_rows[source_path]
                assert master["current_identity"] == canonical
                assert master["existing_split"] == split, (name, record["path"], source_path)
                destination = unique_destination(Path(split) / canonical / Path(record["path"]).name,
                                                 record["sha256"], occupied)
                updated = {**record, **registry.metadata(canonical), "path": str(destination),
                           "master_source_path": master["current_source_path"],
                           "master_match_method": master["match_method"]}
                if name != "source":
                    updated["source_path_before_migration"] = source_path
                after[split].append(updated)
                mapping[record["path"]] = str(destination)
        if not plan["unresolved"]:
            validate_manifest(after, expected_labels=set(registry.identities))
            registry.validate(after)
        path_maps[name] = mapping
        plan["datasets"][name] = {"root": str(root), "manifest_path": str(manifest_path),
                                  "before": before, "after": after, "path_map": mapping,
                                  "manifest_sha256_before": sha(manifest_path)}
        print(f"Planned {name}: {sum(map(len, after.values()))} images; unresolved={len(plan['unresolved'])}", flush=True)
    for name in ("reviewed_crops", "detector_20261012_crops"):
        for records in plan["datasets"][name]["after"].values():
            for record in records:
                record["source_path"] = path_maps["source"][record.pop("source_path_before_migration")]
    detector_paths = sorted(p for p in inventory(DETECTOR) if Path(p).parts[:2] in (("images", "train"), ("images", "val")))
    def detector_record(relative):
        path = DETECTOR / relative
        digest = sha(path)
        candidates = sha_sources.get(digest, set())
        if len(candidates) == 1:
            original = source_details[next(iter(candidates))]
            fp = {"sha256": digest, "pixels_sha256": original["pixels_sha256"], "size": original["size"]}
        else:
            fp = file_fingerprint(path)
            candidates = pixel_sources.get(fp["pixels_sha256"], set())
        if len(candidates) != 1:
            return {"unresolved": relative, "source_candidates": sorted(candidates)}
        source = next(iter(candidates))
        master = master_rows[source]
        label = Path(relative)
        label = Path("labels", *label.parts[1:]).with_suffix(".txt")
        content = (DETECTOR / label).read_bytes()
        boxes = parse_labels(content, description=str(label))
        return {"image": relative, "label": str(label), "detector_split": Path(relative).parts[1],
                "source_path": path_maps["source"][source], "master_source_path": master["current_source_path"],
                "master_match_method": master["match_method"], **registry.metadata(master["current_identity"]),
                **fp, "label_sha256": hashlib.sha256(content).hexdigest(), "num_boxes": len(boxes)}
    # Persist the fully checked source/crop plan before slower detector decoding.
    write(AUDIT / "source_crop_plan.json", plan)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(detector_record, detector_paths), 1):
            if "unresolved" in record:
                plan["unresolved"].append({"dataset": "detector", **record})
            else:
                plan["detector"].append(record)
            if index % 1000 == 0:
                print(f"Linked detector references: {index}/{len(detector_paths)}", flush=True)
    write(AUDIT / "plan.json", plan)
    print(f"Plan saved; {len(plan['unresolved'])} unresolved references", flush=True)
    return plan


def publish(plan):
    if plan["unresolved"]:
        raise ValueError(f"Resolve {len(plan['unresolved'])} references before applying")
    if (AUDIT / "verification.json").exists():
        raise FileExistsError("Already published")
    before = AUDIT / "before"
    before.mkdir(exist_ok=True)
    write(before / "classifier_label_order_156.json", {
        "class_names": sorted({r["id"] for rs in plan["datasets"]["reviewed_crops"]["before"].values() for r in rs}),
        "note": "Original classifier output order; never replace with the canonical 149-name order."})
    shutil.copytree(REPO / "animal-face-id/configs", before / "configs", dirs_exist_ok=True)
    stages = {}
    for name, dataset in plan["datasets"].items():
        root = Path(dataset["root"])
        if sha(dataset["manifest_path"]) != dataset["manifest_sha256_before"]:
            raise ValueError(f"Manifest changed after planning: {name}")
        stage = root.with_name("." + root.name + ".canonical-20261014")
        stage.mkdir(exist_ok=False)
        stages[name] = stage
        for records in dataset["before"].values():
            for record in records:
                old = root / record["path"]
                new = stage / dataset["path_map"][record["path"]]
                new.parent.mkdir(parents=True, exist_ok=True)
                os.link(old, new)
        write(stage / "splits.json", dataset["after"])
        write(stage / "identity_registry.json", plan["registry"])
        write(stage / "split_summary.json", {
            "identity_migration": "2026-10-14", "num_identities": 149,
            "split_assignment_policy": "Existing split membership preserved exactly; no resplit",
            "splits": {s: len(rs) for s, rs in dataset["after"].items()},
            "source_root": str(MASTER), "source_pixel_variant_policy": "Existing image bytes preserved; master references carry match method",
        })
        validate_manifest(dataset["after"])
        assert inventory(stage) == set(dataset["path_map"].values())
        print(f"Staged {name}", flush=True)
    published = []
    try:
        for name, dataset in plan["datasets"].items():
            root = Path(dataset["root"])
            os.rename(root, before / name)
            try:
                os.rename(stages[name], root)
            except BaseException:
                os.rename(before / name, root)
                raise
            published.append(name)
            # Reviewed crops use an external manifest; both active references are canonical.
            if Path(dataset["manifest_path"]) != root / "splits.json":
                shutil.copy2(dataset["manifest_path"], before / (name + "_splits.json"))
                write(dataset["manifest_path"], dataset["after"])
            print(f"Published {name}", flush=True)
    except BaseException:
        for name in reversed(published):
            dataset = plan["datasets"][name]
            root = Path(dataset["root"])
            os.rename(root, stages[name])
            os.rename(before / name, root)
            if Path(dataset["manifest_path"]) != root / "splits.json":
                shutil.copy2(before / (name + "_splits.json"), dataset["manifest_path"])
        raise
    detector_manifest = DETECTOR / "identity_manifest.json"
    write(detector_manifest, {"version": "2026-10-14", "identity_registry": str(REGISTRY),
        "source_root": str(SETS["source"][0]), "master_root": str(MASTER),
        "identity_scope": "Source-folder identity only; class 0 still means macaque_face for every box",
        "records": plan["detector"]})
    write(DETECTOR / "identity_registry.json", plan["registry"])
    with (AUDIT / "reference_updates.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "before_path", "current_path", "preserved_before_path"])
        for name, dataset in plan["datasets"].items():
            for old, new in dataset["path_map"].items():
                writer.writerow([name, str(Path(dataset["root"]) / old), str(Path(dataset["root"]) / new), str(before / name / old)])
    write(AUDIT / "published.json", {"datasets": published, "status": "published_pending_verification"})


def link_legacy_detector():
    """Recover the old export's exact full-path MD5 despite JPEG recompression."""
    from PIL import Image
    plan = read(AUDIT / "source_crop_plan.json")
    originals = {r["path"]: r for r in jsonl(OLD_AUDIT / "fingerprints.jsonl")}
    names = defaultdict(set)
    for retained in read(OLD_AUDIT / "repair_plan.json")["retained"]:
        for old in retained["group_original_paths"]:
            absolute = "/home/ylj20/FacialRecognitionTest/macaque_split_data/" + old
            filename = Path(old).stem + "_" + hashlib.md5(absolute.encode()).hexdigest()[:8] + ".jpg"
            names[filename].add((retained["target"], absolute))
    master_rows = {r["existing_source_path"]: r for r in rows(MASTER_AUDIT / "source_to_existing_splits.csv")}
    registry = IdentityRegistry(REGISTRY)
    paths = sorted(p for p in inventory(DETECTOR) if Path(p).parts[:2] in (("images", "train"), ("images", "val")))
    assert len(paths) == 6630
    def record(relative):
        candidates = names[Path(relative).name]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous original detector filename: {relative}: {candidates}")
        source, exported = next(iter(candidates))
        master = master_rows[source]
        label = Path("labels", *Path(relative).parts[1:]).with_suffix(".txt")
        content = (DETECTOR / label).read_bytes()
        boxes = parse_labels(content, description=str(label))
        with Image.open(DETECTOR / relative) as image:
            size = [image.width, image.height]
        return {"image": relative, "label": str(label), "detector_split": Path(relative).parts[1],
                "source_path": plan["datasets"]["source"]["path_map"][source],
                "master_source_path": master["current_source_path"], "master_match_method": master["match_method"],
                **registry.metadata(master["current_identity"]), "sha256": sha(DETECTOR / relative), "size": size,
                "label_sha256": hashlib.sha256(content).hexdigest(), "num_boxes": len(boxes),
                "detector_link_method": "Original exporter MD5 of full source filepath (git 1f527c8)",
                "original_export_source_path": exported}
    with ThreadPoolExecutor(max_workers=4) as pool:
        plan["detector"] = list(pool.map(record, paths))
    assert not plan["unresolved"]
    write(AUDIT / "plan.json", plan)
    print(f"Exact historical filepath hash linked all {len(paths)} detector photos", flush=True)


def verify(plan):
    registry = IdentityRegistry(REGISTRY)
    report = {"version": "2026-10-14", "identities": len(registry.identities), "datasets": {},
              "images_changed": 0, "split_assignments_changed": 0, "unresolved_references": 0}
    for name, dataset in plan["datasets"].items():
        root = Path(dataset["root"])
        payload = read(dataset["manifest_path"])
        assert payload == dataset["after"] == read(root / "splits.json")
        registry.validate(payload)
        validate_manifest(payload)
        assert inventory(root) == {r["path"] for rs in payload.values() for r in rs}
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(checked_bytes, [(root / r["path"], r) for rs in payload.values() for r in rs]))
        old_to_new = dataset["path_map"]
        after_by_path = {r["path"]: r for rs in payload.values() for r in rs}
        for split, records in dataset["before"].items():
            assert len(records) == len(payload[split])
            for r in records:
                new = after_by_path[old_to_new[r["path"]]]
                assert Path(new["path"]).parts[0] == split
                for field in ("sha256", "pixels_sha256", "mtime_ns", "file_bytes"):
                    assert new[field] == r[field]
                if name != "source":
                    assert (SETS["source"][0] / new["source_path"]).is_file()
                assert (MASTER / new["master_source_path"]).is_file()
        for split in payload:
            folders = {p.name for p in (root / split).iterdir() if p.is_dir()}
            assert folders == set(registry.identities)
        report["datasets"][name] = {"counts": {s: len(rs) for s, rs in payload.items()},
                                   "identities_per_split": {s: len({r['id'] for r in rs}) for s, rs in payload.items()},
                                   "manifest_sha256": sha(dataset["manifest_path"]), "byte_and_RGB_duplicates": 0}
        print(f"Verified canonical {name}", flush=True)
    detector = read(DETECTOR / "identity_manifest.json")["records"]
    assert detector == plan["detector"]
    assert {p for p in inventory(DETECTOR) if Path(p).parts[:2] in (("images", "train"), ("images", "val"))} == {r["image"] for r in detector}
    def verify_detector(r):
        actual = file_fingerprint(DETECTOR / r["image"])
        assert actual["sha256"] == r["sha256"]
        assert sha(DETECTOR / r["label"]) == r["label_sha256"]
        assert (SETS["source"][0] / r["source_path"]).is_file()
        assert registry.resolve(r["id"]) == r["id"]
        assert r["current_group"] == registry.metadata(r["id"])["current_group"]
        return {"image": r["image"], **actual}
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual_detector = list(pool.map(verify_detector, detector))
    for field in ("sha256", "pixels_sha256"):
        assert len({r[field] for r in actual_detector}) == len(detector)
    write(AUDIT / "detector_fingerprints.json", actual_detector)
    report["detector"] = {"image_counts": dict(Counter(r["detector_split"] for r in detector)),
                          "all_image_and_label_hashes_unchanged": True, "linked_references": len(detector)}
    write(AUDIT / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--link-legacy-detector", action="store_true")
    args = parser.parse_args()
    if args.plan:
        make_plan()
    if args.link_legacy_detector:
        link_legacy_detector()
    if args.apply:
        publish(read(AUDIT / "plan.json"))
    if args.verify:
        verify(read(AUDIT / "plan.json"))


if __name__ == "__main__":
    main()
