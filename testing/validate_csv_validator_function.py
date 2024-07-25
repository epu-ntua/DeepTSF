import os
import pandas as pd
from uc2.load_raw_data import read_and_validate_input
# Directory containing the test CSV files
input_dir = "test_csvs"

# CSV files to test
csv_files = {
    "valid_series.csv": {"multiple": False, "format": "long"},
    "valid_multiple_long.csv": {"multiple": True, "format": "long"},
    "valid_multiple_short.csv": {"multiple": True, "format": "short"},
    "empty_single.csv": {"multiple": False, "format": "long"},
    "wrong_col_names_1_single.csv": {"multiple": False, "format": "long"},
    "wrong_col_names_2_single.csv": {"multiple": False, "format": "long"},
    "invalid_date_format_single.csv": {"multiple": False, "format": "long"},
    "non_float_value_single.csv": {"multiple": False, "format": "long"},
    "duplicate_dates_single.csv": {"multiple": False, "format": "long"},
    "non_increasing_dates_single.csv": {"multiple": False, "format": "long"},
    "missing_date_1_single.csv": {"multiple": False, "format": "long"},
    "missing_date_2_single.csv": {"multiple": False, "format": "long"},
    "empty_multiple_long.csv": {"multiple": True, "format": "long"},
    "empty_multiple_short.csv": {"multiple": True, "format": "short"},
    "wrong_col_names_1_multiple_long.csv": {"multiple": True, "format": "long"},
    "wrong_col_names_2_multiple_long.csv": {"multiple": True, "format": "long"},
    "wrong_col_names_1_multiple_short.csv": {"multiple": True, "format": "short"},
    "wrong_col_names_2_multiple_short.csv": {"multiple": True, "format": "short"},
    "no_index_long.csv": {"multiple": True, "format": "long"},
    "no_index_short.csv": {"multiple": True, "format": "short"},
    "wrong_index_long.csv": {"multiple": True, "format": "long"},
    "wrong_index_short.csv": {"multiple": True, "format": "short"},
    "invalid_date_format_multiple_long.csv": {"multiple": True, "format": "long"},
    "invalid_date_format_multiple_short.csv": {"multiple": True, "format": "short"},
    "non_float_value_multiple_long.csv": {"multiple": True, "format": "long"},
    "non_float_value_multiple_short.csv": {"multiple": True, "format": "short"},
    "float_id_value_multiple_long.csv": {"multiple": True, "format": "long"},
    "float_id_value_multiple_short.csv": {"multiple": True, "format": "short"},
    "duplicate_dates_multiple_long.csv": {"multiple": True, "format": "long"},
    "duplicate_dates_multiple_short.csv": {"multiple": True, "format": "short"},
    "non_increasing_dates_multiple_long.csv": {"multiple": True, "format": "long"},
    "non_increasing_dates_multiple_short.csv": {"multiple": True, "format": "short"},
    "missing_date_1_multiple_long.csv": {"multiple": True, "format": "long"},
    "missing_date_1_multiple_short.csv": {"multiple": True, "format": "short"},
    "missing_date_2_multiple_long.csv": {"multiple": True, "format": "long"},
    "missing_date_2_multiple_short.csv": {"multiple": True, "format": "short"},
    "wrong_comp_nos_multiple_long.csv": {"multiple": True, "format": "long"},
    "wrong_comp_nos_multiple_short.csv": {"multiple": True, "format": "short"},
}

def test_read_and_validate_input():
    for filename, params in csv_files.items():
        file_path = os.path.join(input_dir, filename)
        print(f"\nTesting {filename} (multiple={params['multiple']}, format={params['format']})")

        try:
            ts, resolution = read_and_validate_input(
                series_csv=file_path,
                multiple=params["multiple"],
                format=params["format"],
                log_to_mlflow=False
            )
            print(f"Test passed for {filename}")
        except Exception as e:
            print(f"Test failed for {filename}: {e}")

# Include your read_and_validate_input function and other necessary imports and definitions here
# Ensure the read_and_validate_input function is available in the scope
