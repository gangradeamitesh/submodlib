import inspect
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import math
import time

groundData =np.array( [(3,12.5), (5,13.5), (5.5,13.5), (14.5,13.5), (15,13.5), (15.5,13.5),
(4.5,13), (5,13), (5.5,13), (14.5,13), (15,13), (15.5,13),
(4.5,12.5), (5,12.5), (5.5,12.5), (14.5,12.5), (15,12.5), (15.5,12.5),
(4.5,7.5), (5,7.5), (5.5,7.5), (14.5,7.5), (15,7.5), (15.5,7.5),
(4.5,7), (5,7), (5.5,7), (14.5,7), (15,7), (15.5,7),
(4.5,6.5), (5,6.5), (5.5,6.5), (14.5,6.5), (15,6.5), (7.5,10), (12.5,10), (10,12.5), 
(10,7.5), (8,12.5), (8,7.5), (14,12.5), (14,7.5), (4.5, 15.5), (5,9.5), (5,10.5)] )
print("Number of elements in ground set = ", len(groundData))
groundxs = [x[0] for x in groundData]
groundys = [x[1] for x in groundData]

mutlipleQueryData = np.array([(4.5,13.5), (15.5,6.5)])
multiplequeryxs = [x[0] for x in mutlipleQueryData]
multiplequeryys = [x[1] for x in mutlipleQueryData]

mutlipleQueryData2 = np.array([(4.5,13.5), (15.5,11)])
multiplequeryxs2 = [x[0] for x in mutlipleQueryData2]
multiplequeryys2 = [x[1] for x in mutlipleQueryData2]

singleQueryData = np.array([(4.5,13.5)])
singlequeryxs = [x[0] for x in singleQueryData]
singlequeryys = [x[1] for x in singleQueryData]

plt.scatter(groundxs, groundys, s=50, facecolors='none', edgecolors='black', label="Images")
plt.scatter(multiplequeryxs, multiplequeryys, s=50, color='green', label="Queries")
plt.scatter(multiplequeryxs2, multiplequeryys2, s=50, color='green', label="Queries")


def timed_call(fn , *args , **kwargs):
    start = time.time()
    result = fn(*args , **kwargs)
    elapsed = time.perf_counter() - start
    return result , elapsed

def normalize_maximize_output(items, tol=1e-4):
    return sorted(
        [(int(idx), round(float(gain), 4)) for idx, gain in items],
        key=lambda x: x[1],
    )

def compare_classes(obj_a, obj_b, methods, test_cases):
    results = []

    for name in methods:
        fn_a = getattr(obj_a, name)
        fn_b = getattr(obj_b, name)

        if inspect.isroutine(fn_a) and inspect.isroutine(fn_b):
            for args in test_cases.get(name, [()]):
                try:
                    out_a, t_a = timed_call(fn_a , *args)
                except Exception as err_a:
                    out_a = f"RAISED:{type(err_a).__name__}:{err_a}"
                try:
                    out_b,t_b = timed_call(fn_b , *args)
                except Exception as err_b:
                    out_b = f"RAISED:{type(err_b).__name__}:{err_b}"

                match = (
                    out_a == out_b
                    if not isinstance(out_a, float) or not isinstance(out_b, float)
                    else math.isclose(out_a, out_b, rel_tol=0.0, abs_tol=1e-4)
                    )
                results.append(
                    {
                        "method": name,
                        "args": args,
                        "match": match,
                        "output_a": out_a,
                        "output_b": out_b,
                        "time_a": round(t_a*1000,3),
                        "time_b": round(t_b*1000,3),
                    }
                )
        else:
            results.append(
                {
                    "method": name,
                    "args": None,
                    "match": False,
                    "output_a": "not callable",
                    "output_b": "not callable",
                }
            )
    return results

from submodlib import FacilityLocationVariantMutualInformationFunction as LegacyFLVMI
from submodlib  import FacilityLocationVariantMutualInformation as PyFLVMI

legacy = LegacyFLVMI(
    n=46,
    num_queries=2,
    data=groundData,
    queryData=mutlipleQueryData,
    metric="cosine",
    queryDiversityEta=1.0,
)
py_impl = PyFLVMI(
    n=46,
    num_queries=2,
    data=groundData,
    query_data=mutlipleQueryData,
    metric="cosine",
    queryDiversityEta=1.0,
)

methods_to_check = [
    "evaluate",
    "marginalGain",
    "evaluateWithMemoization",
    "marginalGainWithMemoization",
]

test_inputs = {
    "evaluate": [
        (set(),),
        ({0, 5},),
        ({1, 3, 8},),
    ],
    "marginalGain": [
        (set(), 0),
        ({0, 5}, 3),
    ],
    "evaluateWithMemoization": [
        # memo needs to be primed; wrap call sequence inside a helper below
    ],
    "marginalGainWithMemoization": [
        # likewise; use a wrapper that calls setMemoization before measuring
    ],
}

# For methods that require prior memo state, build small wrappers.
def evaluate_with_memo(obj, subset):
    obj.setMemoization(subset)
    return obj.evaluateWithMemoization(subset)

def marginal_with_memo(obj, subset, element):
    obj.setMemoization(subset)
    return obj.marginalGainWithMemoization(subset, element)

# Extend comparison with the memo-aware wrappers.
comparison = []
for entry in compare_classes(
    legacy,
    py_impl,
    ["evaluate", "marginalGain"],
    test_inputs,
):
    comparison.append(entry)

for subset in ({0, 5}, {1, 3, 8}):
    comparison.append(
        {
            "method": "evaluateWithMemoization",
            "args": (subset,),
            "match": evaluate_with_memo(legacy, subset) == evaluate_with_memo(py_impl, subset),
            "output_a": evaluate_with_memo(legacy, subset),
            "output_b": evaluate_with_memo(py_impl, subset),
        }
    )
for subset, element in [({0, 5}, 3), ({1, 3}, 4)]:
    comparison.append(
        {
            "method": "marginalGainWithMemoization",
            "args": (subset, element),
            "match": marginal_with_memo(legacy, subset, element)
            == marginal_with_memo(py_impl, subset, element),
            "output_a": marginal_with_memo(legacy, subset, element),
            "output_b": marginal_with_memo(py_impl, subset, element),
        }
    )

legacy_max = normalize_maximize_output(
    legacy.maximize(
        budget=10,
        optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        epsilon=0.0,
        verbose=False,
        show_progress=False,
        costs=None,
        costSensitiveGreedy=False,
    )
)

python_max = normalize_maximize_output(
    py_impl.maximize(
        budget=10,
        optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        epsilon=0.0,
        verbose=False,
        show_progress=False,
        costs=None,
        costSensitiveGreedy=False,
    )
)

legacy_max, t_a = timed_call(legacy.maximize,  budget=10,
        optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        epsilon=0.0,
        verbose=False,
        show_progress=False,
        costs=None,
        costSensitiveGreedy=False,)
python_max, t_b = timed_call(py_impl.maximize , budget=10,
        optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        epsilon=0.0,
        verbose=False,
        show_progress=False,
        costs=None,
        costSensitiveGreedy=False,)
comparison.append(
    {
        "method": "maximize",
        "args": ("budget=10",),  # free-form note
        "match": normalize_maximize_output(legacy_max) == normalize_maximize_output(python_max),
        "output_a": normalize_maximize_output(legacy_max),
        "output_b": normalize_maximize_output(python_max),
        "time_a_ms": round(t_a * 1000, 3),
        "time_b_ms": round(t_b * 1000, 3),

    }
)

pprint(comparison)
