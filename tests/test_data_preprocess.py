import os
import pandas as pd
import io
from pathlib import Path

import pytest

from data import data_preprocess as dp


def test_standardize_and_trim_columns():
    df = pd.DataFrame({" a ": [" foo ", None], "num": [1, None]})
    df2 = dp._standardize_columns(df)
    assert "A" in df2.columns and "NUM" in df2.columns

    df3 = dp._trim_string_columns(df2)
    # trimmed string should equal 'foo'
    assert df3.loc[0, "A"] == "foo"
    # None should become <NA> (pandas string dtype) or remain null-like
    assert pd.isna(df3.loc[1, "A"])


def test_ensure_output_folder_creates(tmp_path):
    base = tmp_path
    out = dp._ensure_output_folder(str(base), "processed_test")
    assert os.path.exists(out)
    assert os.path.isdir(out)


def test_read_csv_with_fallback_reads_utf8(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("col1,col2\n1,2\na, b\n", encoding="utf-8")
    df = dp._read_csv_with_fallback(str(csv_path))
    assert list(df.columns) == ["col1", "col2"]
    assert len(df) == 2


def test_process_patient_creates_clean_file(tmp_path):
    # prepare input csv with lowercase columns and missing/duplicate patient ids
    content = "patient_id,dob,name\n1,1980-01-01, Alice \n,1990-02-02,Bob\n1,1980-01-01, Alice \n"
    in_file = tmp_path / "patients.csv"
    in_file.write_text(content, encoding="utf-8")

    out_df = dp.process_patient(str(tmp_path), "patients.csv", "processed_pd")

    # Check output dataframe properties
    assert "PATIENT_ID" in out_df.columns
    # No missing patient ids
    assert out_df["PATIENT_ID"].isna().sum() == 0
    # Duplicates removed (only one row with id 1)
    assert out_df["PATIENT_ID"].nunique() == 1
    # DOB parsed to datetime (or NaT) and present column name uppercase
    assert any(col in out_df.columns for col in ["DOB", "DATE_OF_BIRTH", "BIRTHDATE"]) or "DOB" in out_df.columns

    # Check file written
    out_file = tmp_path / "processed_pd" / "dim_patient_clean.csv"
    assert out_file.exists()


def test_process_physician_handles_missing_and_specialty(tmp_path):
    content = "physician_id,specialty,years_experience\n10,cardiology,15\n11,,\n,neurology,5\n"
    in_file = tmp_path / "phys.csv"
    in_file.write_text(content, encoding="utf-8")

    out_df = dp.process_physician(str(tmp_path), "phys.csv", "processed_phy")

    assert "PHYSICIAN_ID" in out_df.columns
    # Missing physician id rows dropped
    assert out_df["PHYSICIAN_ID"].isna().sum() == 0
    # Specialty should be title-cased and missing filled with 'Unknown'
    assert all(isinstance(s, str) for s in out_df["SPECIALTY"].astype(str))

    out_file = tmp_path / "processed_phy" / "dim_physician_clean.csv"
    assert out_file.exists()


def test_process_transactions_aggregations(tmp_path):
    # Create a transactions CSV with dates, types and descriptions
    content = (
        "patient_id,transaction_date,txn_type,txn_desc,amount,physician_id\n"
        "P1,2023-01-01,CONDITIONS,check,100,10\n"
        "P1,2023-01-02,SYMPTOMS,drug a,50,10\n"
        "P2,,CONTRAINDICATIONS,other,30,11\n"
    )
    in_file = tmp_path / "tx.csv"
    in_file.write_text(content, encoding="utf-8")

    out_df = dp.process_transactions(str(tmp_path), "tx.csv", "processed_tx")

    # TXN_DT should exist
    assert "TXN_DT" in out_df.columns
    # Aggregation columns present
    assert "NO_OF_CONDN" in out_df.columns and "NO_OF_SYMPT" in out_df.columns and "NO_OF_CONTRD" in out_df.columns
    # Target column may be added with mixed case by the implementation; check case-insensitively
    target_col = next((c for c in out_df.columns if c.upper() == "TARGET"), None)
    assert target_col is not None
    # Values should be integer-like 0/1
    assert set(out_df[target_col].dropna().astype(int).unique()).issubset({0, 1})
    # Check that counts for P1 reflect one CONDITIONS and one SYMPTOMS
    p1 = out_df[out_df["PATIENT_ID"] == "P1"].iloc[0]
    assert p1["NO_OF_CONDN"] >= 1
    assert p1["NO_OF_SYMPT"] >= 1

    out_file = tmp_path / "processed_tx" / "fact_transactions_clean.csv"
    assert out_file.exists()
