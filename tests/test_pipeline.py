"""Owner: E. Feedback rows from production are cleaned before retraining."""

import numpy as np
import pytest

pytest.importorskip("prefect")

from dropout.data import TARGET, load_data  # noqa: E402
from dropout.features import FEATURES  # noqa: E402
from dropout.pipeline import combine_with_feedback  # noqa: E402


def test_no_feedback_returns_none():
    assert combine_with_feedback(None) is None


def test_incomplete_feedback_rows_are_dropped():
    feedback = load_data().head(5)[FEATURES + [TARGET]].copy()
    feedback.loc[feedback.index[0], TARGET] = None
    feedback.loc[feedback.index[1], "Admission grade"] = np.nan
    cleaned = combine_with_feedback(feedback)
    assert len(cleaned) == 3
    assert list(cleaned.columns) == FEATURES + [TARGET]
