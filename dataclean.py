import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from typing import Optional, Union
import logging


class DataClean:
    def __init__(
        self,
        detect_binary: bool = True,
        numeric_dtype: bool = True,
        one_hot: bool = True,
        na_strategy: str = "mean",
        normalize: bool = True,
        datetime_columns: list[str] = [],
        remove_columns: list[str] = [],
        verbose: bool = True,
    ):
        """
        Initialize DataClean with various cleaning options.

        detect_binary: If True, detect and convert binary columns to 0 and 1.
        numeric_dtype: If True, attempt to convert columns to numeric dtypes when possible.
        one_hot: If True, perform one-hot encoding on categorical columns.
        na_strategy: Strategy for handling NA values. Options: "mean", "median", "most_frequent", "remove_row".
        normalize: If True, normalize non-binary numeric columns.
        datetime_columns: List of column names containing datetime data.
        remove_columns: List of column names to remove from the dataset.
        verbose: If True, print progress and information during cleaning.
        """
        self.detect_binary = detect_binary
        self.numeric_dtype = numeric_dtype
        self.one_hot = one_hot
        self.na_strategy = na_strategy
        self.normalize = normalize
        self.datetime_columns = datetime_columns
        self.remove_columns = remove_columns
        self.verbose = verbose
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("DataClean")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def clean(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Clean the input DataFrame based on the initialized settings.

        df: Input Pandas DataFrame to clean.
        is_training: If True, fit transformers on this data. If False, use pre-fitted transformers.
        :return: Cleaned Pandas DataFrame.
        """
        self.logger.info("Starting data cleaning process...")

        # Make a copy of the DataFrame to avoid modifying the original
        df = df.copy()

        # Remove unwanted columns
        df = self._remove_columns(df)

        # Convert datetime columns
        df = self._convert_datetime(df)

        # Detect and convert binary columns
        if self.detect_binary:
            df = self._detect_binary(df)

        # Convert to numeric dtypes where possible
        if self.numeric_dtype:
            df = self._convert_numeric(df)

        # Handle NA values
        df = self._handle_na(df, is_training)

        # Perform one-hot encoding
        if self.one_hot:
            df = self._one_hot_encode(df)

        # Normalize non-binary numeric columns
        if self.normalize:
            df = self._normalize(df, is_training)

        self.logger.info("Data cleaning process completed.")
        return df

    def _remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove specified columns from the DataFrame."""
        for col in self.remove_columns:
            if col in df.columns:
                df = df.drop(columns=col)
                self.logger.info(f"Removed column: {col}")
        return df

    def _convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert specified columns to datetime dtype."""
        for col in self.datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                self.logger.info(f"Converted column to datetime: {col}")
        return df

    def _detect_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert binary columns to 0 and 1."""
        for col in df.columns:
            if df[col].nunique() == 2:
                unique_values = df[col].unique()
                df[col] = df[col].map({unique_values[0]: 0, unique_values[1]: 1})
                self.logger.info(f"Converted binary column: {col}")
        return df

    def _convert_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to convert columns to numeric dtypes."""
        for col in df.columns:
            if col not in self.datetime_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                    self.logger.info(f"Converted column to numeric: {col}")
                except ValueError:
                    pass  # Column can't be converted to numeric
        return df

    def _handle_na(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Handle NA values based on the specified strategy."""
        if self.na_strategy == "remove_row":
            df = df.dropna()
            self.logger.info("Removed rows with NA values")
        else:
            if is_training:
                self.imputer = SimpleImputer(strategy=self.na_strategy)
                df = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
            else:
                df = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
            self.logger.info(f"Handled NA values using strategy: {self.na_strategy}")
        return df

    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform one-hot encoding on categorical columns."""
        categorical_columns = df.select_dtypes(include=["object"]).columns
        df = pd.get_dummies(df, columns=categorical_columns)
        self.logger.info("Performed one-hot encoding on categorical columns")
        return df

    def _normalize(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Normalize non-binary numeric columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if df[col].nunique() > 2]

        if is_training:
            self.scaler = StandardScaler()
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        else:
            df[numeric_columns] = self.scaler.transform(df[numeric_columns])

        self.logger.info("Normalized non-binary numeric columns")
        return df


def main():
    parser = argparse.ArgumentParser(description="Improved Data Cleaner")
    parser.add_argument("input_file", type=str, help="Input CSV file path")
    parser.add_argument("--output_file", type=str, help="Output CSV file path")
    parser.add_argument(
        "--detect_binary", action="store_true", help="Detect and convert binary columns"
    )
    parser.add_argument(
        "--numeric_dtype",
        action="store_true",
        help="Convert to numeric dtypes where possible",
    )
    parser.add_argument(
        "--one_hot", action="store_true", help="Perform one-hot encoding"
    )
    parser.add_argument(
        "--na_strategy",
        type=str,
        default="mean",
        choices=["mean", "median", "most_frequent", "remove_row"],
        help="Strategy for handling NA values",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize non-binary numeric columns"
    )
    parser.add_argument(
        "--datetime_columns", nargs="+", help="List of datetime column names"
    )
    parser.add_argument("--remove_columns", nargs="+", help="List of columns to remove")
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress information"
    )

    args = parser.parse_args()

    # Read input CSV
    df = pd.read_csv(args.input_file)

    # Initialize and run the data cleaner
    cleaner = DataClean(
        detect_binary=args.detect_binary,
        numeric_dtype=args.numeric_dtype,
        one_hot=args.one_hot,
        na_strategy=args.na_strategy,
        normalize=args.normalize,
        datetime_columns=args.datetime_columns or [],
        remove_columns=args.remove_columns or [],
        verbose=args.verbose,
    )

    cleaned_df = cleaner.clean(df)

    # Save or display the results
    if args.output_file:
        cleaned_df.to_csv(args.output_file, index=False)
        print(f"Cleaned data saved to {args.output_file}")
    else:
        print(cleaned_df)


if __name__ == "__main__":
    main()
