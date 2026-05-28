import re
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Medical abbreviation normalisation map
ABBREVIATION_MAP = {
    r"\bHBA1C\b": "HbA1c",
    r"\bGLYCOHEMOGLOBIN\b": "HbA1c",
    r"\bGLYCOSYLATED HEMOGLOBIN\b": "HbA1c",
    r"\bBLOOD PRESSURE\b": "BP",
    r"\bCHOL\b": "Cholesterol Total",
    r"\bTOTAL CHOLESTEROL\b": "Cholesterol Total",
    r"\bWHITE BLOOD CELL\b": "WBC",
    r"\bWHITE BLOOD CELLS\b": "WBC",
    r"\bW.B.C\b": "WBC",
    r"\bRED BLOOD CELL\b": "RBC",
    r"\bRED BLOOD CELLS\b": "RBC",
    r"\bR.B.C\b": "RBC",
    r"\bPLATELETS\b": "Platelet",
    r"\bPLT\b": "Platelet",
    r"\bHEMOGLOBIN\b": "Hemoglobin",
    r"\bHGB\b": "Hemoglobin",
    r"\bHB\b": "Hemoglobin",
    r"\bCREAT\b": "Creatinine",
    r"\bSERUM CREATININE\b": "Creatinine",
}

def clean_ocr_text(text: str) -> str:
    """Clean OCR errors, excessive whitespaces, and normalise characters."""
    if not text:
        return ""
    
    # 1. Standardise line endings and spacing
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)  # Collapse tabs and multiple spaces
    
    # 2. Fix typical OCR errors in numeric contexts (e.g., "1O0 mg/dL", "O.8 mg/dL", "1.2l")
    # Fix l or I to 1 when directly adjacent to digits or a decimal point
    text = re.sub(r"(?<=\d)[lI]\b", "1", text)
    text = re.sub(r"\b[lI](?=\d)", "1", text)
    # Fix O or o to 0 when directly adjacent to digits or a decimal point
    text = re.sub(r"(?<=\d)[Oo]\b", "0", text)
    text = re.sub(r"\b[Oo](?=\d)", "0", text)
    text = re.sub(r"(?<=\d\.)[Oo]\b", "0", text)
    
    # 3. Normalise medical abbreviations
    for pattern, replacement in ABBREVIATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    return text

def extract_numerical_values(text: str) -> Dict[str, Tuple[float, str]]:
    """
    Scans the text for medical parameters and extracts their numerical values and units.
    Returns: {test_name: (value, unit)}
    """
    extracted = {}
    cleaned = clean_ocr_text(text)
    
    # Custom regex search patterns for each medical test
    patterns = {
        "Glucose": [
            r"(?:Glucose|Fasting Glucose|Blood Glucose|GLU)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mg/dl|mmol/L|mmol/l)?\b",
        ],
        "HbA1c": [
            r"(?:HbA1c|A1c|Glycated Hemoglobin)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(%)\b",
        ],
        "Creatinine": [
            r"(?:Creatinine|Serum Creatinine|CREAT)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mg/dl|umol/L|umol/l)?\b",
        ],
        "WBC": [
            r"(?:WBC|White Blood Cell|White Blood Cells)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(?:x\s*10\^3|x10\^3|k|thousand)?\s*(?:/\s*[uμ]L|/\s*mm3)?\b",
        ],
        "Hemoglobin": [
            r"(?:Hemoglobin|Hgb|Hb)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(g/dL|g/dl|g/L|g/l)?\b",
        ],
        "Platelet": [
            r"(?:Platelet|Platelets|PLT)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(?:x\s*10\^3|x10\^3|k|thousand)?\s*(?:/\s*[uμ]L|/\s*mm3)?\b",
        ],
        "Cholesterol Total": [
            r"(?:Cholesterol Total|Total Cholesterol|CHOL)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mg/dl|mmol/L|mmol/l)?\b",
        ],
        "LDL": [
            r"(?:LDL|LDL-C|LDL Cholesterol)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mg/dl|mmol/L|mmol/l)?\b",
        ],
        "HDL": [
            r"(?:HDL|HDL-C|HDL Cholesterol)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mg/dl|mmol/L|mmol/l)?\b",
        ]
    }

    # First, let's extract Blood Pressure specifically as it has a systolic/diastolic format (e.g. 120/80 mmHg)
    bp_match = re.search(r"(?:BP|Blood Pressure|B\.P\.)\s*[:=-]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(mmHg|mm Hg)?\b", cleaned, re.IGNORECASE)
    if bp_match:
        try:
            systolic = float(bp_match.group(1))
            diastolic = float(bp_match.group(2))
            unit = bp_match.group(3) or "mmHg"
            extracted["Systolic BP"] = (systolic, unit)
            extracted["Diastolic BP"] = (diastolic, unit)
        except Exception as e:
            logger.warning(f"Failed to parse blood pressure matches: {str(e)}")

    # Check for independent Systolic/Diastolic labels
    if "Systolic BP" not in extracted:
        sys_match = re.search(r"(?:Systolic|Systolic BP)\s*[:=-]?\s*(\d{2,3})\s*(mmHg|mm Hg)?\b", cleaned, re.IGNORECASE)
        if sys_match:
            extracted["Systolic BP"] = (float(sys_match.group(1)), sys_match.group(2) or "mmHg")
            
    if "Diastolic BP" not in extracted:
        dia_match = re.search(r"(?:Diastolic|Diastolic BP)\s*[:=-]?\s*(\d{2,3})\s*(mmHg|mm Hg)?\b", cleaned, re.IGNORECASE)
        if dia_match:
            extracted["Diastolic BP"] = (float(dia_match.group(1)), dia_match.group(2) or "mmHg")

    # Run regex patterns for standard single values
    for test, pattern_list in patterns.items():
        for pat in pattern_list:
            match = re.search(pat, cleaned, re.IGNORECASE)
            if match:
                try:
                    val_str = match.group(1)
                    val = float(val_str)
                    try:
                        unit = match.group(2) or ""
                    except IndexError:
                        unit = ""
                    
                    # Normalize WBC and Platelets if they are expressed in thousands (e.g., "6.5" instead of "6500", "250" instead of "250000")
                    if test == "WBC":
                        if val < 50.0:  # e.g., 4.5 or 11.0 means 4500 or 11000
                            val = val * 1000.0
                            unit = "/μL"
                    elif test == "Platelet":
                        if val < 1000.0:  # e.g., 150 or 400 means 150000 or 400000
                            val = val * 1000.0
                            unit = "/μL"
                            
                    extracted[test] = (val, unit)
                    break  # Stop at first matching regex for this test
                except Exception as e:
                    logger.warning(f"Error parsing match for {test}: {str(e)}")

    return extracted
