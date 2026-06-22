"""Cross-check for rolling-window data leakage.

Trains the SAME model on the SAME windows/features using two split strategies:
  1. random window split  (old approach -> windows from one .mat file land in BOTH train and test)
  2. grouped split by .mat file (new approach -> a file's windows are entirely train OR entirely test)

A large accuracy gap between (1) and (2) is the signature of rolling-window leakage.
Also reports a 3rd, even stricter split: held-out by bearing CODE (e.g. all KA01 windows test-only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pinn_motor_fault.features import BearingPhysics, PhysicsFeatureExtractor
from pinn_motor_fault.model import PhysicsInformedNN
from pinn_motor_fault.paderborn import (
    BEARING_CODE_RE,
    infer_label_from_path,
    load_paderborn_windows,
)
from pinn_motor_fault.train import stratified_group_split, stratified_split


def code_of(source: str) -> str:
    for part in reversed(Path(source).parts):
        m = BEARING_CODE_RE.search(part.upper())
        if m:
            return m.group(1)
    return "UNK"


def stratified_code_split(labels, sources, test_fraction=0.25, seed=23):
    """Hold out entire bearing CODES (e.g. KA01) so no bearing instance is shared."""
    rng = np.random.default_rng(seed)
    codes = np.array([code_of(s) for s in sources])
    code_label = {c: labels[np.flatnonzero(codes == c)[0]] for c in set(codes)}
    train_codes, test_codes = set(), set()
    for lab in sorted(set(code_label.values())):
        group = [c for c, l in code_label.items() if l == lab]
        rng.shuffle(group)
        k = max(1, int(round(len(group) * test_fraction)))
        test_codes.update(group[:k])
        train_codes.update(group[k:])
    train = [i for i, c in enumerate(codes) if c in train_codes]
    test = [i for i, c in enumerate(codes) if c in test_codes]
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def evaluate(name, features, labels, physics, train_idx, test_idx, sources, epochs):
    model = PhysicsInformedNN(input_dim=features.shape[1], hidden_dim=48,
                              physics_weight=0.20, learning_rate=0.015)
    model.fit(features[train_idx], labels[train_idx], physics[train_idx],
              epochs=epochs, batch_size=64, verbose=False)
    _, train_acc = model.loss_and_accuracy(features[train_idx], labels[train_idx], physics[train_idx])
    _, test_acc = model.loss_and_accuracy(features[test_idx], labels[test_idx], physics[test_idx])
    shared = set(sources[i] for i in train_idx) & set(sources[i] for i in test_idx)
    shared_codes = set(code_of(sources[i]) for i in train_idx) & set(code_of(sources[i]) for i in test_idx)
    print(f"\n=== {name} ===")
    print(f"  train windows={train_idx.size}  test windows={test_idx.size}")
    print(f"  .mat files shared between train&test : {len(shared)}")
    print(f"  bearing codes shared between train&test: {len(shared_codes)}")
    print(f"  train_accuracy={train_acc:.4f}  test_accuracy={test_acc:.4f}")
    return test_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/paderborn/extracted"))
    ap.add_argument("--codes", nargs="*", default=None,
                    help="Subset of bearing-code folders to use; default = all.")
    ap.add_argument("--max-windows-per-file", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    if args.codes:
        dirs = [args.data_dir / c for c in args.codes]
    else:
        dirs = [args.data_dir]

    all_w, all_l, all_s = [], [], []
    for d in dirs:
        w, l, s = load_paderborn_windows(
            d, window_size=4096, stride=2048,
            signal_preference="vibration", max_windows_per_file=args.max_windows_per_file,
        )
        all_w.append(w); all_l.append(l); all_s.extend(s)
    windows = np.concatenate(all_w); labels = np.concatenate(all_l); sources = all_s

    print(f"Loaded {windows.shape[0]} windows from {len(set(sources))} .mat files "
          f"across {len(set(code_of(s) for s in sources))} bearing codes.")
    print("Class distribution:", {c: int(np.sum(labels == c)) for c in sorted(set(labels))})

    batch = PhysicsFeatureExtractor(BearingPhysics()).transform(windows, sources)
    feats, phys = batch.features, batch.physics_targets

    # 1. random window split (leaky old approach)
    tr, te = stratified_split(labels, test_fraction=0.25, seed=11)
    a_random = evaluate("RANDOM window split (old approach, leakage possible)",
                        feats, labels, phys, tr, te, sources, args.epochs)

    # 2. grouped-by-file split (new approach)
    tr, te = stratified_group_split(labels, sources, test_fraction=0.25, seed=17)
    a_group = evaluate("GROUPED-BY-FILE split (current report approach)",
                       feats, labels, phys, tr, te, sources, args.epochs)

    # 3. held-out bearing CODE split (strictest)
    tr, te = stratified_code_split(labels, sources, test_fraction=0.25, seed=23)
    a_code = evaluate("HELD-OUT BEARING-CODE split (strictest)",
                      feats, labels, phys, tr, te, sources, args.epochs)

    print("\n================ SUMMARY ================")
    print(f"  random-window split test acc : {a_random:.4f}")
    print(f"  grouped-by-file split test acc: {a_group:.4f}")
    print(f"  held-out-code split test acc  : {a_code:.4f}")
    print(f"  leakage gap (random - grouped): {a_random - a_group:+.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
