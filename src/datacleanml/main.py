"""
from datacleanml import DataCleanML
import pandas as pd

df = pd.read_csv('your_data.csv')
cleaner = DataCleanML(detect_binary=True, normalize=True)
cleaned_df = cleaner.clean(df)

python datacleanml.py input_file.csv --output_filename cleaned_data.csv --detect_binary --normalize --verbose
"""

import argparse
import logging
import os
from typing import Any, Union

import dateutil.parser
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm


class DataCleanML:
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
            'remove_correlated': False,
            'correlation_threshold': 0.95,
            'skip_normalize': [],
            'extract_date_info': True,
        }

        self.config.update(config)
        self.verbose = verbose
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("DataCleanML")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def clean(self, df: pd.DataFrame, is_training: bool = True, save_path: str = None) -> pd.DataFrame:
        self.logger.info(f"Starting data cleaning process...")

        # Input validation
        self._validate_input(df, is_training)

        # Define operations
        operations = [
            (self._remove_columns, True, "Remove specified columns"),
            (self._convert_datetime, True, "Convert datetime columns"),
            (self._detect_binary, self.config['detect_binary'], "Detect and convert binary columns"),
            (self._convert_numeric, self.config['numeric_dtype'], "Convert to numeric dtypes"),
            (self._handle_outliers, self.config['handle_outliers'], "Handle outliers"),
            (self._handle_na, True, "Handle NA values"),
            (self._one_hot_encode, self.config['one_hot'], "Perform one-hot encoding"),
            (self._normalize, self.config['normalize'], "Normalize non-binary numeric columns"),
            (self._feature_engineering, True, "Perform feature engineering"),
            (self._select_features, self.config['feature_selection'] and is_training, "Select features"),
            (self._remove_correlated_features, self.config['remove_correlated'], "Remove correlated features"),
            (self._extract_date_info, self.config['extract_date_info'], "Extract date info"),
        ]

        performed_operations = []
        excluded_operations = []

        # Use tqdm for progress tracking if verbose
        for operation, condition, description in tqdm(operations, disable=not self.verbose):
            if condition:
                try:
                    df = operation(df, is_training)
                    performed_operations.append(description)
                except Exception as e:
                    self.logger.error(f"Error in {operation.__name__}: {str(e)}")
                    raise
            else:
                excluded_operations.append(description)

        self.logger.info("Data cleaning process completed.")
        self.logger.info(f"Performed operations: {', '.join(performed_operations)}")
        self.logger.info(f"Excluded operations: {', '.join(excluded_operations)}")

        if save_path:
            df.to_csv(save_path, index=False)
            self.logger.info(f"Cleaned data saved to {save_path}")

        return df

    def _validate_input(self, df: pd.DataFrame, is_training: bool) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Ensure datetime_columns is a list
        datetime_columns = self.config['datetime_columns']
        if isinstance(datetime_columns, str):
            datetime_columns = [datetime_columns]
        elif not isinstance(datetime_columns, list):
            datetime_columns = list(datetime_columns)

        unused_datetime_cols = set(datetime_columns) - set(df.columns)
        if unused_datetime_cols:
            self.logger.warning("Specified datetime column(s) not in the DataFrame: %s", unused_datetime_cols)

        # Update the config with the list version
        self.config['datetime_columns'] = datetime_columns

    def _remove_columns(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Remove specified columns from the DataFrame."""
        for col in self.config['remove_columns']:
            if col in df.columns:
                df = df.drop(columns=col)
                self.logger.info(f"Removed column: {col}")
            else:
                self.logger.warning(f"Requested column for removal not found: {col}")
        return df

    def _convert_datetime_manual(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Convert specified columns to datetime dtype.
        DEPRECATED
        """
        for col in self.config['datetime_columns']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                self.logger.info(f"Converted column to datetime: {col}")
        return df

    def _convert_datetime(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Automatically detect and convert datetime columns."""
        for col in df.columns:
            if not is_datetime64_any_dtype(df[col]):
                # Check if the column contains string data
                if df[col].dtype == 'object':
                    try:
                        # Try to parse a sample of non-null values
                        sample = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()
                        if all(self._is_date(val) for val in sample):
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            self.logger.info(f"Automatically converted column to datetime: {col}")
                    except (ValueError, TypeError):
                        # If conversion fails, skip this column
                        pass
        return df

    @staticmethod
    def _is_date(string):
        """Check if a string can be parsed as a date."""
        try:
            dateutil.parser.parse(string)
            return True
        except (ValueError, TypeError):
            return False

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
            if col not in self.config['datetime_columns']:
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
        changed_columns = []
        unchanged_columns = []

        if self.config['na_strategy'] == "remove_row":
            initial_rows = len(df)
            df = df.dropna()
            rows_removed = initial_rows - len(df)
            if rows_removed > 0:
                self.logger.info(f"Removed {rows_removed} rows with NA values")
                changed_columns = df.columns.tolist()
            else:
                self.logger.info("No rows with NA values found")
                unchanged_columns = df.columns.tolist()
        else:
            # Separate datetime columns
            datetime_columns = df.select_dtypes(include=['datetime64']).columns
            non_datetime_columns = df.columns.difference(datetime_columns)

            # Handle non-datetime columns
            for column, strategy in self.config['column_specific_imputation'].items():
                if column in non_datetime_columns:
                    if df[column].isnull().any():
                        if is_training:
                            imputer = SimpleImputer(strategy=strategy)
                            df[[column]] = imputer.fit_transform(df[[column]])
                        else:
                            df[[column]] = imputer.transform(df[[column]])
                        self.logger.info(f"Imputed column {column} using strategy: {strategy}")
                        changed_columns.append(column)
                    else:
                        unchanged_columns.append(column)

            # Handle remaining non-datetime columns with global strategy
            remaining_columns = [
                col for col in non_datetime_columns if col not in self.config['column_specific_imputation']
            ]
            if remaining_columns:
                columns_with_na = df[remaining_columns].columns[df[remaining_columns].isnull().any()].tolist()
                if columns_with_na:
                    if is_training:
                        self.global_imputer = SimpleImputer(strategy=self.config['na_strategy'])
                        df[remaining_columns] = self.global_imputer.fit_transform(df[remaining_columns])
                    else:
                        df[remaining_columns] = self.global_imputer.transform(df[remaining_columns])
                    self.logger.info(
                        f"Handled NA values for remaining non-datetime columns ({columns_with_na}) using strategy: {self.config['na_strategy']}"
                    )
                    changed_columns.extend(columns_with_na)
                unchanged_columns.extend([col for col in remaining_columns if col not in columns_with_na])

            # Handle datetime columns
            for column in datetime_columns:
                if df[column].isnull().any():
                    if is_training:
                        # For datetime columns, use forward fill, then backward fill
                        df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
                    else:
                        df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
                    self.logger.info(f"Imputed datetime column {column} using forward and backward fill")
                    changed_columns.append(column)
                else:
                    unchanged_columns.append(column)

        self.logger.info(f"Columns which had NA: {', '.join(changed_columns) if changed_columns else 'None'}")
        self.logger.info(
            f"Columns which didn't have NA: {', '.join(unchanged_columns) if unchanged_columns else 'None'}"
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
        """Normalize non-binary numeric columns, except those specified in skip_normalize."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if df[col].nunique() > 2]

        # Exclude columns specified in skip_normalize
        columns_to_normalize = [col for col in numeric_columns if col not in self.config['skip_normalize']]

        if columns_to_normalize:
            if is_training:
                self.scaler = StandardScaler()
                df[columns_to_normalize] = self.scaler.fit_transform(df[columns_to_normalize])
            else:
                df[columns_to_normalize] = self.scaler.transform(df[columns_to_normalize])

            self.logger.info(f"Normalized non-binary numeric columns: {', '.join(columns_to_normalize)}")

        if self.config['skip_normalize']:
            self.logger.info(f"Skipped normalization for columns: {', '.join(self.config['skip_normalize'])}")

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

    def _remove_correlated_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Remove highly correlated features."""
        if not is_training:
            return df

        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > self.config['correlation_threshold'])
        ]

        if to_drop:
            df = df.drop(to_drop, axis=1)
            self.logger.info(f"Removed {len(to_drop)} highly correlated features: {', '.join(to_drop)}")
        else:
            self.logger.info("No highly correlated features found to remove.")

        return df

    def _extract_date_info(self, df: pd.DataFrame, is_training: bool):
        # Convert 'Date' column to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week

        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)

        # Drop the original 'Date' column
        # df = df.drop(['Date'], axis=1)

        return df


def main():
    parser = argparse.ArgumentParser(description="Data Cleaner")
    parser.add_argument("input_file", type=str, help="Input CSV file path")
    parser.add_argument("--output_filename", type=str, help="Output CSV file path")
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
    parser.add_argument("--handle_outliers", action="store_true")
    parser.add_argument("--datetime_columns", nargs="+", help="List of datetime column names")
    parser.add_argument("--remove_columns", nargs="+", help="List of columns to remove")
    parser.add_argument("--remove_correlated", action="store_true", help="Remove highly correlated features")
    parser.add_argument(
        "--correlation_threshold",
        type=float,
        default=0.95,
        help="Threshold for correlation above which features will be removed",
    )
    parser.add_argument("--skip_normalize", nargs="+", help="List of columns to skip during normalization")
    parser.add_argument("--extract_date_info", action="store_true")
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
        'remove_correlated': args.remove_correlated,
        'correlation_threshold': args.correlation_threshold,
        'skip_normalize': args.skip_normalize or [],
    }

    # Initialize and run the data cleaner
    cleaner = DataCleanML(config=config, verbose=args.verbose)

    cleaned_df = cleaner.clean(df)

    # Save or display the results
    if args.output_filename:
        output_file = args.output_filename
    else:
        input_file_name = os.path.basename(args.input_file)
        input_file_base, input_file_ext = os.path.splitext(input_file_name)
        output_file = f"{input_file_base}_cleaned{input_file_ext}"
        output_file = os.path.join(os.path.dirname(args.input_file), output_file)

    # Save the results
    cleaned_df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")


if __name__ == "__main__":
    main()
