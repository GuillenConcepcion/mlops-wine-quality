"""
Unit tests for data loading, ingestion, and stratification.
"""

import pytest
import pandas as pd
from src.config import config, TargetMode
from src.data.loader import load_raw_wine_data, get_train_test_split, transform_target


def test_load_raw_wine_data():
    """Verify total unified dataset size, columns, and absence of missing values."""
    df = load_raw_wine_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6497
    assert "is_red" in df.columns
    assert "quality" in df.columns
    assert df["is_red"].isin([0, 1]).all()
    assert df.isnull().sum().sum() == 0


def test_train_test_split_stratification():
    """Verify stratified split preserves proportional target distribution."""
    X_train, X_test, y_train, y_test = get_train_test_split()

    assert len(X_train) == 5197
    assert len(X_test) == 1300
    assert len(X_train) + len(X_test) == 6497

    # Train and test must share the primary target classes
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    assert {5, 6, 7}.issubset(train_classes)
    assert {5, 6, 7}.issubset(test_classes)


def test_target_transformation_modes():
    """Verify target transformation for multiclass and segmented modes."""
    s = pd.Series([3, 5, 6, 7, 8, 9])

    # Multiclass
    y_multi = transform_target(s, mode=TargetMode.MULTICLASS)
    assert list(y_multi) == [3, 5, 6, 7, 8, 9]

    # Segmented 3-class (Low: <=5, Med: 6, High: >=7)
    y_seg = transform_target(s, mode=TargetMode.SEGMENTED_3CLASS)
    assert list(y_seg) == [0, 0, 1, 2, 2, 2]

    # Binary Premium (Standard: <7, Premium: >=7)
    y_bin = transform_target(s, mode=TargetMode.BINARY_PREMIUM)
    assert list(y_bin) == [0, 0, 0, 1, 1, 1]
