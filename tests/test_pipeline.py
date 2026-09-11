"""
Unit tests for custom transformers and pipeline components.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.transformers import EnologicalFeatureEngineer


def test_enological_feature_engineer_transformation():
    """Verify calculated enological metrics are generated cleanly without NaNs."""
    raw_sample = pd.DataFrame([{
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.36,
        "residual_sugar": 20.7,
        "chlorides": 0.045,
        "free_sulfur_dioxide": 45.0,
        "total_sulfur_dioxide": 170.0,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
        "is_red": 0
    }])

    fe = EnologicalFeatureEngineer()
    out_df = fe.fit_transform(raw_sample)

    assert "acidity_ratio" in out_df.columns
    assert "free_so2_ratio" in out_df.columns
    assert "bound_so2" in out_df.columns
    assert "total_acidity" in out_df.columns
    assert "alcohol_sugar_ratio" in out_df.columns
    assert "sulphates_chlorides_ratio" in out_df.columns

    # Verify math
    assert np.isclose(out_df["acidity_ratio"].iloc[0], 0.27 / 7.0, atol=1e-4)
    assert np.isclose(out_df["bound_so2"].iloc[0], 170.0 - 45.0, atol=1e-4)
    assert np.isclose(out_df["total_acidity"].iloc[0], 7.0 + 0.27 + 0.36, atol=1e-4)
    assert out_df.isnull().sum().sum() == 0
