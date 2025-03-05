import numpy as np
import pandas as pd

from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)

class EmptyDataframe(Exception):
    """
    Exception raised if dataframe is empty.
    """
    def __init__(self, from_database):
        super().__init__("Dataframe provided is too short (empty or with only 1 row)" + (" or does not exist in mongo database" if from_database else ""))

class EmptySeries(Exception):
    """
    Exception raised if dataframe has empty series.
    """
    def __init__(self):
        super().__init__("Dataframe provided has empty series")

class DatetimesNotInOrder(Exception):
    """
    Exception raised if dates in series_csv are not sorted.
    """
    def __init__(self, first_wrong_date, row_id, ts_id=None, id=None):
        if ts_id == None:
            self.message = f"Datetimes in series_csv are not sorted. First unordered date: {first_wrong_date} in row with id {row_id}."
        else:
            self.message = f"Datetimes are not sorted for component with id {id} of timeseries {ts_id}. First unordered date: {first_wrong_date} in row with id {row_id}."
        super().__init__(self.message)

class WrongColumnNames(Exception):
    """
    Exception raised if series_csv has wrong column names.
    """
    def __init__(self, columns, col_num, names, format="single"):
        names = ", ".join(names)
        if format == "short":
            self.message = f'Column names provided: {columns}. For {format} format, series_csv must have at least {col_num} columns named {names} in any order.'
        elif format == "long":
            self.message = f'Column names provided: {columns}. For {format} format, series_csv must have {col_num} columns named {names} in any order.'
        else:
            self.message = f'Column names provided: {columns}. For single time series, series_csv must have {col_num} columns named {names}.'
        super().__init__(self.message)

class WrongDateFormat(Exception):
    """Exception raised for errors in the input date format."""
    def __init__(self, invalid_date, id_row):
        self.message = f"Date format must be 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD'. First invalid date: {invalid_date} in row with id {id_row}"
        super().__init__(self.message)

class DuplicateDateError(Exception):
    """Exception raised for duplicate dates in the series."""
    def __init__(self, duplicate_date, row_id, ts_id=None, id=None):
        if ts_id == None:
            self.message = f"Timeseries can not have any duplicate dates. First duplicate: {duplicate_date} in row with id {row_id}"
        else:
            self.message = f"Component {id} of timeseries {ts_id} has duplicate dates. First duplicate: {duplicate_date} in row with id {row_id}"

        super().__init__(self.message)

class CountryDoesNotExist(Exception):
    """
    Exception raised if the country specified does not exist/have holidays.
    """
    def __init__(self):
        super().__init__("The country specified does not exist/have holidays")

class WrongIDs(Exception):
    """
    Exception raised if the IDs present in a multiple timeseries file are not consecutive integers.
    """
    def __init__(self, ids):
        self.message = f'ID names provided: {ids}. IDs in a multiple timeseries file must be consecutive integers.'
        super().__init__(self.message)

class DifferentComponentDimensions(Exception):
    """
    Exception raised if not all timeseries in a multiple timeseries file have the same number of components.
    """
    def __init__(self, comp_dict):
        self.message = f'Not all timeseries in multiple timeseries file have the same number of components:\n'
        self.message += "\n".join(f"Timeseries {key} has {value} component(s)" for key, value in comp_dict.items())
        super().__init__(self.message)

class NanInSet(Exception):
    """
    Exception raised if val or test set has nan values.
    """
    def __init__(self):
        self.message = f'Validation and test set can not have any nan values'
        super().__init__(self.message)

class MandatoryArgNotSet(Exception):
    """
    Exception raised if a mandatory argument was not set by the user.
    """
    def __init__(self, argument_name, mandatory_prerequisites):
        if mandatory_prerequisites:
            mandatory_prerequisites = "\n".join(("- " + args[0] + "=" + args[1]) for args in mandatory_prerequisites)
            self.message = f'Argument {argument_name} is mandatory since the following conditions apply: \n{mandatory_prerequisites}.\nIt was set to None / not set.'
        else:
            self.message = f'Argument {argument_name} is mandatory and set to None / not set.'
        super().__init__(self.message)

class NotValidConfig(Exception):
    """
    Exception raised if config is not an entrypoint in config file and not a valid json string.
    """
    def __init__(self):
        self.message = f'config is not an entrypoint in config file or a valid json string'
        super().__init__(self.message)

class NoUpsamplingException(Exception):
    """
    Exception raised if the user tries to convert a series to a lower resolution.
    """
    def __init__(self):
        self.message = f'Upsampling is not allowed. Change the target resolution of the series'
        super().__init__(self.message)

class TsUsedIdDoesNotExcist(Exception):
    """
    Exception raised if ts_used_id chosen by the user does not exist in the multiple time series file.
    """
    def __init__(self):
        self.message = f'This ts_used_id does not exist in the multiple time series file'
        super().__init__(self.message)


class DifferentFrequenciesMultipleTS(Exception):
    """
    Exception raised if multiple / multivariate series file has different inferred resolutions.
    """
    def __init__(self, infered_resolution_1, id_1, ts_id_1, infered_resolution_2, id_2, ts_id_2):
        self.message = f'Resolution of 2 components has been inferred to be different: ts {ts_id_1}, component {id_1}, resolution {infered_resolution_1}, and ts {ts_id_2}, component {id_2}, resolution {infered_resolution_2}'
        super().__init__(self.message)

class EvalSeriesNotFound(Exception):
    """
    Exception raised if eval_series parameter is not found.
    """
    def __init__(self, eval_series):
        self.message = f"eval_series parameter '{eval_series}' not found in file"
        super().__init__(self.message)

class NonIntegerMultipleIndexError(Exception):
    """Exception raised when the index of a multiple series is not of integer type."""
    def __init__(self, index_dtype):
        self.message = f"The index is not of integer type. The current index type is: {index_dtype}"
        super().__init__(self.message)

class MissingMultipleIndexError(Exception):
    """Exception raised when the index of multuple series format is missing values."""
    def __init__(self, missing_index):
        self.message = f"The index is missing values. The first missing index is: {missing_index}"
        super().__init__(self.message)

class ComponentTooShortError(Exception):
    """Exception raised if a component of the timeseries is too short (length less than or equal to 1)."""
    def __init__(self, length, ts_id, id):
        self.message = f"Component {id} of timeseries {ts_id} is too short (length = {length})"
        super().__init__(self.message)

class TSIDNotFoundInferenceError(Exception):
    """Exception raised when ts_id_pred is not found in ts_id_l of a multiple ts during inference."""
    def __init__(self, ts_id_pred, stored):
        self.ts_id_pred = ts_id_pred
        self.message = f"Time series id (ts_id_pred) {ts_id_pred} not found in"
        if stored: 
            self.message += " stored ts_id_l which the model was trained on"
        else:
            self.message += " multiple series file provided by the user"
        super().__init__(self.message)