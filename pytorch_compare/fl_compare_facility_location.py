"""
Quick comparison between the PyTorch and C++/Python Facility Location implementations.

Runs a small random experiment and checks:
1) evaluate() values on the same random subset
2) NaiveGreedy selections and gains
"""

import numpy as np
import torch

from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.submodlib_pytorch.regular_functions.facility_location import (
    FacilityLocation as TorchFacilityLocation,
)


def compare_once(n=20, d=64, budget=5, seed=0, metric="cosine", atol=1e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    data_np = rng.standard_normal(size=(n, d), dtype=np.float32)
    data_torch = torch.tensor(data_np, dtype=torch.float32)

    fl_cpp = FacilityLocationFunction(
        n=n,
        mode="dense",
        separate_rep=False,
        data=data_np,
        metric=metric,
        create_dense_cpp_kernel_in_python=True,
    )
    fl_torch = TorchFacilityLocation(n=n, data=data_torch, metric=metric, device="cpu")

    subset = set(rng.choice(n, size=min(5, n), replace=False).tolist())
    eval_cpp = fl_cpp.evaluate(subset)
    eval_torch = fl_torch.evaluate(subset)

    greedy_cpp = fl_cpp.maximize(
        budget=budget,
        optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    greedy_torch = fl_torch.maximize(
        budget=budget,
        optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
        show_progress=False,
    )

    indices_cpp = [x[0] for x in greedy_cpp]
    indices_torch = [x[0] for x in greedy_torch]
    gains_cpp = [x[1] for x in greedy_cpp]
    gains_torch = [x[1] for x in greedy_torch]

    print("Subset:", subset)
    print(f"evaluate   cpp={eval_cpp:.6f}  torch={eval_torch:.6f}  diff={abs(eval_cpp - eval_torch):.6f}")
    print("Greedy indices cpp :", indices_cpp)
    print("Greedy indices torch:", indices_torch)
    print("Greedy gains   cpp :", gains_cpp)
    print("Greedy gains   torch:", gains_torch)

    if abs(eval_cpp - eval_torch) > atol:
        print(f"[WARN] evaluate mismatch > {atol}")
    if indices_cpp != indices_torch:
        print("[WARN] greedy index order differs")
    if any(abs(a - b) > atol for a, b in zip(gains_cpp, gains_torch)):
        print(f"[WARN] greedy gains mismatch > {atol}")


if __name__ == "__main__":
    compare_once()
