import json
import math

import numpy as np

from prism_model.export_staff import _clean


def test_clean_makes_reports_json_safe():
    out = _clean({"a": np.float64(1.5), "b": float("nan"), "c": [np.int64(3), (1, 2)], "d": {"e": np.inf}})
    assert out == {"a": 1.5, "b": None, "c": [3, [1, 2]], "d": {"e": None}}
    json.dumps(out, allow_nan=False)  # would raise on NaN/inf
    assert not any(isinstance(v, float) and math.isnan(v) for v in out.values() if v is not None)
