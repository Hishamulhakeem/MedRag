import pytest
from backend.services.preprocessing import extract_numerical_values
from backend.services.abnormality import analyze_abnormalities

def test_extract_numerical_values():
    sample_text = """
    Patient Report
    Fasting Glucose: 112 mg/dL
    Total Cholesterol = 210 mg/dl
    WBC: 6.5 x10^3/uL
    Hemoglobin: 12.8 g/dL
    Blood Pressure: 135/85 mmHg
    """
    vals = extract_numerical_values(sample_text)
    
    assert "Glucose" in vals
    assert vals["Glucose"][0] == 112.0
    
    assert "Cholesterol Total" in vals
    assert vals["Cholesterol Total"][0] == 210.0
    
    assert "WBC" in vals
    assert vals["WBC"][0] == 6500.0  # Normalized (6.5 * 1000)
    
    assert "Hemoglobin" in vals
    assert vals["Hemoglobin"][0] == 12.8
    
    assert "Systolic BP" in vals
    assert vals["Systolic BP"][0] == 135.0
    
    assert "Diastolic BP" in vals
    assert vals["Diastolic BP"][0] == 85.0

def test_analyze_abnormalities_classification():
    # Setup test input
    extracted = {
        "Glucose": (135.0, "mg/dL"),        # High (Normal 70-100, Borderline <=125)
        "WBC": (2500.0, "/uL"),             # Low (Normal 4500-11000)
        "HbA1c": (10.5, "%"),               # Critical (>= 10.0)
        "Platelet": (220000.0, "/uL"),      # Normal (150k - 400k)
        "HDL": (35.0, "mg/dL")              # Low (< 40)
    }
    
    results = analyze_abnormalities(extracted)
    results_map = {r["test_name"]: r for r in results}
    
    assert results_map["Glucose"]["status"] == "HIGH"
    assert results_map["WBC"]["status"] == "LOW"
    assert results_map["HbA1c"]["status"] == "CRITICAL"
    assert results_map["Platelet"]["status"] == "NORMAL"
    assert results_map["HDL"]["status"] == "LOW"
