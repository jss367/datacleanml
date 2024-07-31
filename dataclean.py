"""
from dataclean import DataClean
import pandas as pd

df = pd.read_csv('your_data.csv')
cleaner = DataClean(detect_binary=True, normalize=True)
cleaned_df = cleaner.clean(df)

python datacleaner.py input_file.csv --output_file cleaned_data.csv --detect_binary --normalize --verbose
"""

import argparse
import logging
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm


class DataClean:
    def __init__(self, config: dict[str, Union[bool, str, list[str], dict[str, Any]]] = {}, verbose: bool = True):
        self.config = {
            'detect_binary': True,
            'numeric_dtype': True,
            'one_hot': True,
            'na_strategy': 'mean',
            'normalize': True,
            'datetime_columns': [],
            'remove_columns': [],
            'column_specific_imputation': {},
            'feature_engineering': {'polynomial_features': [], 'bin_columns': []},
            'handle_outliers': False,
            'advanced_imputation': False,
            'feature_selection': False,
            'n_features_to_select': 10,
        }
        self.config.update(config)
        self.verbose = verbose
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("DataClean")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def clean(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        self.logger.info("Starting data cleaning process...")

        # Input validation
        self._validate_input(df)

        # Define operations
        operations = [
            (self._remove_columns, True),
            (self._convert_datetime, True),
            (self._detect_binary, self.config['detect_binary']),
            (self._convert_numeric, self.config['numeric_dtype']),
            (self._handle_outliers, self.config['handle_outliers']),
            (self._handle_na, True),
            (self._one_hot_encode, self.config['one_hot']),
            (self._normalize, self.config['normalize']),
            (self._feature_engineering, True),
            (self._select_features, self.config['feature_selection'] and is_training),
        ]

        # Use tqdm for progress tracking if verbose
        for operation, condition in tqdm(operations, disable=not self.verbose):
            if condition:
                try:
                    df = operation(df, is_training)
                except Exception as e:
                    self.logger.error(f"Error in {operation.__name__}: {str(e)}")
                    raise

        self.logger.info("Data cleaning process completed.")
        return df

    def _validate_input(self, df: pd.DataFrame, is_training: bool) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if is_training and 'target' not in df.columns:
            self.logger.warning("Target column not found. Some operations may be skipped.")
        if set(self.config['datetime_columns']).difference(df.columns):
            self.logger.warning("Some specified datetime columns are not in the DataFrame")

    def _remove_columns(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Remove specified columns from the DataFrame."""
        for col in self.remove_columns:
            if col in df.columns:
                df = df.drop(columns=col)
                self.logger.info(f"Removed column: {col}")
        return df

    def _convert_datetime(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Convert specified columns to datetime dtype."""
        for col in self.datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                self.logger.info(f"Converted column to datetime: {col}")
        return df

    def _detect_binary(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Detect and convert binary columns to 0 and 1."""
        for col in df.columns:
            if df[col].nunique() == 2:
                unique_values = df[col].unique()
                df[col] = df[col].map({unique_values[0]: 0, unique_values[1]: 1})
                self.logger.info(f"Converted binary column: {col}")
        return df

    def _convert_numeric(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Attempt to convert columns to numeric dtypes."""
        for col in df.columns:
            if col not in self.datetime_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                    self.logger.info(f"Converted column to numeric: {col}")
                except ValueError:
                    pass  # Column can't be converted to numeric
        return df

    def _handle_outliers(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Handle outliers using Isolation Forest."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(df[numeric_columns])
        df = df[outliers != -1]
        self.logger.info("Removed outliers using Isolation Forest")
        return df

    def _handle_na(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        if self.config['na_strategy'] == "remove_row":
            df = df.dropna()
            self.logger.info("Removed rows with NA values")
        elif self.config['advanced_imputation']:
            if is_training:
                self.imputer = IterativeImputer(random_state=42)
                df = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
            else:
                df = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
            self.logger.info("Handled NA values using advanced imputation")
        else:
            for column, strategy in self.config['column_specific_imputation'].items():
                if column in df.columns:
                    if is_training:
                        imputer = SimpleImputer(strategy=strategy)
                        df[[column]] = imputer.fit_transform(df[[column]])
                    else:
                        df[[column]] = imputer.transform(df[[column]])
                    self.logger.info(f"Imputed column {column} using strategy: {strategy}")

            # Handle remaining columns with global strategy
            remaining_columns = [col for col in df.columns if col not in self.config['column_specific_imputation']]
            if remaining_columns:
                if is_training:
                    self.global_imputer = SimpleImputer(strategy=self.config['na_strategy'])
                    df[remaining_columns] = self.global_imputer.fit_transform(df[remaining_columns])
                else:
                    df[remaining_columns] = self.global_imputer.transform(df[remaining_columns])
                self.logger.info(
                    f"Handled NA values for remaining columns using strategy: {self.config['na_strategy']}"
                )
        return df

    def _select_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        if 'target' not in df.columns:
            self.logger.warning("Target column not found. Skipping feature selection.")
            return df

        correlations = df.corr()['target'].abs().sort_values(ascending=False)
        selected_features = correlations.nlargest(self.config['n_features_to_select'] + 1).index.tolist()
        selected_features.remove('target')
        df = df[selected_features + ['target']]
        self.logger.info(
            f"Selected top {self.config['n_features_to_select']} features based on correlation with target"
        )
        return df

    def _one_hot_encode(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
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

    def _feature_engineering(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        # Implement polynomial features
        poly_columns = self.config['feature_engineering']['polynomial_features']
        if poly_columns:
            if is_training:
                self.poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = self.poly.fit_transform(df[poly_columns])
            else:
                poly_features = self.poly.transform(df[poly_columns])

            feature_names = self.poly.get_feature_names_out(poly_columns)
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
            df = pd.concat([df, poly_df], axis=1)
            self.logger.info(f"Added polynomial features for columns: {poly_columns}")

        # Implement binning
        for column in self.config['feature_engineering']['bin_columns']:
            if column in df.columns:
                df[f"{column}_binned"] = pd.qcut(df[column], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                self.logger.info(f"Added binned feature for column: {column}")

        return df


def main():
    parser = argparse.ArgumentParser(description="Improved Data Cleaner")
    parser.add_argument("input_file", type=str, help="Input CSV file path")
    parser.add_argument("--output_file", type=str, help="Output CSV file path")
    parser.add_argument("--detect_binary", action="store_true", help="Detect and convert binary columns")
    parser.add_argument(
        "--numeric_dtype",
        action="store_true",
        help="Convert to numeric dtypes where possible",
    )
    parser.add_argument("--one_hot", action="store_true", help="Perform one-hot encoding")
    parser.add_argument(
        "--na_strategy",
        type=str,
        default="mean",
        choices=["mean", "median", "most_frequent", "remove_row"],
        help="Strategy for handling NA values",
    )
    parser.add_argument("--normalize", action="store_true", help="Normalize non-binary numeric columns")
    parser.add_argument("--datetime_columns", nargs="+", help="List of datetime column names")
    parser.add_argument("--remove_columns", nargs="+", help="List of columns to remove")
    parser.add_argument("--verbose", action="store_true", help="Print progress information")

    args = parser.parse_args()

    # Read input CSV
    df = pd.read_csv(args.input_file)

    config = {
        'detect_binary': args.detect_binary,
        'numeric_dtype': args.numeric_dtype,
        'one_hot': args.one_hot,
        'na_strategy': args.na_strategy,
        'normalize': args.normalize,
        'datetime_columns': args.datetime_columns or [],
        'remove_columns': args.remove_columns or [],
    }

    # Read input CSV
    df = pd.read_csv(args.input_file)

    # Initialize and run the data cleaner
    cleaner = DataClean(config=config, verbose=args.verbose)

    cleaned_df = cleaner.clean(df)

    # Save or display the results
    if args.output_file:
        cleaned_df.to_csv(args.output_file, index=False)
        print(f"Cleaned data saved to {args.output_file}")
    else:
        print(cleaned_df)


if __name__ == "__main__":
    main()
