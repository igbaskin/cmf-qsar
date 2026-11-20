#!/usr/bin/env python
"""
The neural device on the splits of plan item ND1: its predictions and its learned descriptors.

"The device" is the QSPR Neural Device (Baskin, Palyulin, Zefirov, J. Chem. Inf. Comput. Sci. 37
(1997) 715; package ``neurodevice``, https://github.com/igbaskin/qspr-neural-device).  It is a
separate project, and this script is the only place where plan item ND1 runs it.  It does not
import ``cmfqsar``, so it runs in the Python environment of the device, which may have a GPU
that the environment of this program has not.

For one table of molecules (``name,smiles,label``), one method of the device's own evaluation
protocol and one seed, it

* makes the three-way split of that protocol (random, 70 % training, 15 % validation, 15 % test);
* trains the device as its own benchmarks train it (``benchmarks/protocol.py`` of the device
  project: the sensors chosen on the training rows only, early stopping on the validation rows);
* writes, for **every** molecule of the table, the prediction and the collector signals, which
  are the descriptors the device has learned, with the three index sets.

``benchmarks/run_featuriser.py`` reads these files and uses the same index sets for every other
model, so that the device, the fields and their combinations are scored on the same test rows.

Usage
-----
    python benchmarks/featuriser_device.py --device-root C:/path/to/the/device/checkout \\
        --table benchmarks/cache/featuriser/compas-1D.csv --method mp --seed 0 \\
        --out benchmarks/results/featuriser/device/compas-1D_mp_0.npz [--gpu]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

METHODS = ("oneshot", "mp")
VALID_FRACTION = TEST_FRACTION = 0.15


def run(device_root, table, method, seed, out, gpu=False):
    sys.path.insert(0, os.path.abspath(device_root))
    from benchmarks import protocol as P                      # the device project's protocol
    from benchmarks.splits import make_split
    from neurodevice.core.session import Session

    t0 = time.time()
    session = Session()
    session.load_dataset(table, property_name="label", name_tag="name")
    full = session.dataset
    mols, names, y = list(full.mols), list(full.names), np.asarray(full.y, float)
    train, valid, test = (np.asarray(v, int) for v in make_split(
        mols, "random", int(seed), VALID_FRACTION, TEST_FRACTION))
    # the session holds the training and validation rows only, as protocol.build_session does
    is_train = np.zeros(len(mols), dtype=bool)
    is_train[train] = True
    full.is_train = is_train
    session.dataset = full.subset(np.sort(np.concatenate([train, valid])))
    session._survey = None
    session.invalidate()
    held = P.Held(session, [mols[i] for i in test], y[test], n_train=int(train.size),
                  n_valid=int(valid.size))
    valid_metrics, test_metrics = P.run_method(held, P.METHODS[method], int(seed), use_gpu=gpu)
    pred = np.asarray(session.predict_mols(mols), float)
    collectors = np.asarray(session.collectors_for(mols), float)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out, names=np.array(names), y=y.astype(np.float32), pred=pred.astype(np.float32),
        collectors=collectors.astype(np.float32), train=train.astype(np.int32),
        valid=valid.astype(np.int32), test=test.astype(np.int32),
        method=method, seed=int(seed), seconds=time.time() - t0,
        test_r2=float(test_metrics.r2), test_rmse=float(test_metrics.rmse),
        valid_rmse=float(valid_metrics.rmse))
    print(f"{os.path.basename(out)}: {len(mols)} molecules, {collectors.shape[1]} collectors, "
          f"test R2 {test_metrics.r2:+.3f}, RMSE {test_metrics.rmse:.3f}, "
          f"{time.time() - t0:.0f} s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device-root", required=True, help="the checkout of the device project")
    ap.add_argument("--table", required=True, help="CSV with the columns name, smiles, label")
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true", help="use the PyTorch backend on a GPU")
    a = ap.parse_args(argv)
    run(a.device_root, a.table, a.method, a.seed, a.out, a.gpu)


if __name__ == "__main__":
    main()
