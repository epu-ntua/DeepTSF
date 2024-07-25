import os

# Directory to save the CSV files
output_dir = "test_csvs"
os.makedirs(output_dir, exist_ok=True)

# Data for each CSV
csv_data = {
    "valid_series.csv": """Datetime,Value
2023-01-01 00:00:00,1.0
2023-01-01 01:00:00,2.0
2023-01-01 02:00:00,3.0
2023-01-01 03:00:00,4.0
2023-01-01 04:00:00,5.0
""",
    "valid_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 01:00:00,0,A,2.0
2,2023-01-01 00:00:00,1,B,3.0
3,2023-01-01 01:00:00,1,B,4.0
""",
    "valid_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,2023-01-01,1,A,1.0,2.0
1,2023-01-01,2,B,3.0,4.0
""",
 "empty_single.csv": """Datetime,Value
""",
 "wrong_col_names_1_single.csv": """DT,Value
01-01-2023 00:00:00,1.0
01-01-2023 01:00:00,2.0
01-01-2023 02:00:00,3.0
""",
,
 "wrong_col_names_2_single.csv": """Datetime,Temp
01-01-2023 00:00:00,1.0
01-01-2023 01:00:00,2.0
01-01-2023 02:00:00,3.0
""",
    "invalid_date_format_single.csv": """Datetime,Value
01-01-2023 00:00:00,1.0
01-01-2023 01:00:00,2.0
01-01-2023 02:00:00,3.0
""",
    "non_float_value_single.csv": """Datetime,Value
2023-01-01 00:00:00,1.0
2023-01-01 01:00:00,abc
2023-01-01 02:00:00,3.0
""",
    "duplicate_dates_single.csv": """Datetime,Value
2023-01-01 00:00:00,1.0
2023-01-01 01:00:00,2.0
2023-01-01 01:00:00,3.0
""",
    "non_increasing_dates_single.csv": """Datetime,Value
2023-01-01 00:00:00,1.0
2023-01-01 02:00:00,2.0
2023-01-01 01:00:00,3.0
""",
    "missing_date_1_single.csv": """Datetime,Value
2023-01-01 00:00:00,1.0
2023-01-01 02:00:00,3.0
2023-01-01 03:00:00,4.0
2023-01-01 04:00:00,5.0
""",
    "missing_date_2_single.csv": """Datetime,Value
2023-01-01 00:00:00,1.0
2023-01-01 01:00:00,
2023-01-01 02:00:00,3.0
2023-01-01 03:00:00,4.0
2023-01-01 04:00:00,5.0
""",
    "empty_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
""",
    "empty_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
""",
    "wrong_col_names_1_multiple_long.csv": """,DT,ID,Timeseries ID,Value
0,01-01-2023 00:00:00,0,A,1.0
1,01-01-2023 01:00:00,0,A,2.0
2,01-01-2023 00:00:00,1,B,3.0
3,01-01-2023 01:00:00,1,B,4.0
""",
    "wrong_col_names_2_multiple_long.csv": """,Datetime,ID,TS_ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 01:00:00,0,A,2.0
2,2023-01-01 00:00:00,1,B,3.0
3,2023-01-01 01:00:00,1,B,4.0
""",
    "wrong_col_names_1_multiple_short.csv": """,Date,Ident,Timeseries ID,00:00,12:00
0,2023-01-01,1,A,1.0,2.0
1,2023-01-01,2,B,3.0,4.0
""",
    "wrong_col_names_2_multiple_short.csv": """,Date,ID,TS_ID,00:00,12:00
0,2023-01-01,1,A,1.0,2.0
1,2023-01-01,2,B,3.0,4.0
""",
    "no_index_long.csv": """Datetime,ID,Timeseries ID,Value
2023-01-01 00:00:00,0,A,1.0
2023-01-01 01:00:00,0,A,2.0
2023-01-01 00:00:00,1,B,3.0
2023-01-01 01:00:00,1,B,4.0
""",
    "no_index_short.csv": """Date,ID,Timeseries ID,00:00,12:00
2023-01-01,1,A,1.0,2.0
2023-01-01,2,B,3.0,4.0
""",
    "wrong_index_long.csv": """Datetime,ID,Timeseries ID,Value
0, 2023-01-01 00:00:00,0,A,1.0
1, 2023-01-01 01:00:00,0,A,2.0
3, 2023-01-01 00:00:00,1,B,3.0
4, 2023-01-01 01:00:00,1,B,4.0
""",
    "wrong_index_short.csv": """Date,ID,Timeseries ID,00:00,12:00
3, 2023-01-01,1,A,1.0,2.0
5, 2023-01-01,2,B,3.0,4.0
""",
    "invalid_date_format_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,01-01-2023 00:00:00,0,A,1.0
1,01-01-2023 01:00:00,0,A,2.0
2,01-01-2023 00:00:00,1,B,3.0
3,01-01-2023 01:00:00,1,B,4.0
""",
    "invalid_date_format_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,01-01-2023,1,A,1.0,2.0
1,01-01-2023,2,B,3.0,4.0
""",
    "non_float_value_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 01:00:00,0,A,abc
2,2023-01-01 00:00:00,1,B,3.0
3,2023-01-01 01:00:00,1,B,4.0
""",
    "non_float_value_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,2023-01-01,1,A,1.0,abc
1,2023-01-01,2,B,3.0,4.0
""",
    "float_id_value_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 01:00:00,0.0,A,2.0
2,2023-01-01 00:00:00,1,B,3.0
3,2023-01-01 01:00:00,1,B,4.0
""",
    "float_id_value_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,2023-01-01,1.0,A,1.0,abc
1,2023-01-01,2,B,3.0,4.0
""",
    "duplicate_dates_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 01:00:00,0,A,2.0
2,2023-01-01 01:00:00,1,B,3.0
3,2023-01-01 01:00:00,1,B,4.0
""",
    "duplicate_dates_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,2023-01-01,1,A,1.0,2.0
1,2023-01-01,2,B,3.0,4.0
2,2023-01-01,1,A,5.0,6.0
""",
    "non_increasing_dates_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 02:00:00,0,A,2.0
2,2023-01-01 01:00:00,1,B,3.0
3,2023-01-01 00:00:00,1,B,4.0
""",
    "non_increasing_dates_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,2023-01-01,1,A,1.0,2.0
1,2023-01-02,2,B,3.0,4.0
2,2023-01-01,2,B,5.0,6.0
""",
    "missing_date_1_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 02:00:00,0,A,3.0
2,2023-01-01 03:00:00,1,B,4.0
3,2023-01-01 04:00:00,1,B,5.0
""",
    "missing_date_1_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,2023-01-01,1,A,1.0,2.0
1,2023-01-03,1,A,3.0,4.0
""",
    "missing_date_2_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 01:00:00,0,A,
2,2023-01-01 02:00:00,1,B,3.0
3,2023-01-01 03:00:00,1,B,4.0
""",
    "missing_date_2_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,2023-01-01,1,A,1.0,2.0
1,2023-01-01,2,B,3.0,
2,2023-01-02,1,A,5.0,6.0
""",
    "wrong_comp_nos_multiple_long.csv": """,Datetime,ID,Timeseries ID,Value
0,2023-01-01 00:00:00,0,A,1.0
1,2023-01-01 01:00:00,0,A,2.0
2,2023-01-01 00:00:00,1,B,3.0
3,2023-01-01 01:00:00,1,B,4.0
4,2023-01-01 00:00:00,0,B,3.0
5,2023-01-01 01:00:00,0,B,4.0

""",
    "wrong_comp_nos_multiple_short.csv": """,Date,ID,Timeseries ID,00:00,12:00
0,2023-01-01,1,A,1.0,2.0
1,2023-01-01,2,B,3.0,4.0
2,2023-01-01,1,B,3.0,4.0
""",
}

# Write each CSV file
for filename, data in csv_data.items():
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(data)

print(f"CSV files have been created in the directory '{output_dir}'")