"""
Data Loader & Ingestion Pipeline.
Handles automated fetching from UCI repository, ingestion of Red and White wine datasets,
unification with 'is_red' indicator, and strictly stratified train/test splitting.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.config import config, TargetMode, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def download_uci_dataset(url: str, dest_path: Path) -> Path:
    """Download a file from a URL if it does not already exist."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Dataset already cached at: {dest_path}")
        return dest_path

    logger.info(f"Downloading dataset from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Downloaded {dest_path.stat().st_size} bytes successfully.")
    return dest_path


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case without quotes or spaces."""
    clean_cols = (
        df.columns.str.strip()
        .str.replace('"', "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )
    df.columns = clean_cols
    return df


def load_raw_wine_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load and unify both Red and White wine datasets from UCI.
    Adds the 'is_red' binary flag (1 for Red, 0 for White).
    """
    red_path = RAW_DATA_DIR / "winequality-red.csv"
    white_path = RAW_DATA_DIR / "winequality-white.csv"

    if force_download or not red_path.exists():
        download_uci_dataset(config.uci_red_wine_url, red_path)
    if force_download or not white_path.exists():
        download_uci_dataset(config.uci_white_wine_url, white_path)

    # Ingest Red Wine
    df_red = pd.read_csv(red_path, sep=";")
    df_red = standardize_column_names(df_red)
    df_red["is_red"] = 1
    logger.info(f"Loaded Red Wine samples: {len(df_red)}")

    # Ingest White Wine
    df_white = pd.read_csv(white_path, sep=";")
    df_white = standardize_column_names(df_white)
    df_white["is_red"] = 0
    logger.info(f"Loaded White Wine samples: {len(df_white)}")

    # Unify Datasets
    df_unified = pd.concat([df_red, df_white], ignore_index=True)
    logger.info(f"Total unified samples: {len(df_unified)} (Features: {df_unified.shape[1]})")

    # Persist unified dataset to processed directory
    unified_path = PROCESSED_DATA_DIR / "winequality_unified.parquet"
    df_unified.to_parquet(unified_path, index=False)
    logger.info(f"Unified dataset persisted to: {unified_path}")

    return df_unified


def transform_target(y: pd.Series, mode: TargetMode = TargetMode.MULTICLASS) -> pd.Series:
    """
    Transforms the continuous/discrete target into configured classification targets.
    - MULTICLASS: Raw quality score (integer 3 to 9)
    - SEGMENTED_3CLASS: 0 = Low (<=5), 1 = Medium (6), 2 = High (>=7)
    - BINARY_PREMIUM: 0 = Standard (<7), 1 = Premium (>=7)
    """
    if mode == TargetMode.MULTICLASS:
        return y.astype(int)
    elif mode == TargetMode.SEGMENTED_3CLASS:
        return pd.cut(
            y,
            bins=[-float("inf"), 5, 6, float("inf")],
            labels=[0, 1, 2],
        ).astype(int)
    elif mode == TargetMode.BINARY_PREMIUM:
        return (y >= 7).astype(int)
    else:
        raise ValueError(f"Unsupported TargetMode: {mode}")


def get_train_test_split(
    df: Optional[pd.DataFrame] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    target_mode: Optional[TargetMode] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs strict stratified train/test split.
    Guarantees no data leakage before feature transformation.
    """
    if df is None:
        df = load_raw_wine_data()

    test_size = test_size if test_size is not None else config.test_size
    random_state = random_state if random_state is not None else config.random_state
    target_mode = target_mode if target_mode is not None else config.target_mode

    X = df.drop(columns=[config.target_column])
    y_raw = df[config.target_column]
    y = transform_target(y_raw, mode=target_mode)

    # For stratifying on extremely rare classes (e.g. quality 9 has only 5 samples),
    # ensure minimum samples per class for StratifiedKFold or group singletons if needed.
    class_counts = y.value_counts()
    logger.info(f"Class distribution:\n{class_counts.sort_index().to_dict()}")

    # Stratified split based on target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(f"Split complete: Train set={len(X_train)} samples, Test set={len(X_test)} samples.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_raw_wine_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print("Ingestion verification passed successfully.")
