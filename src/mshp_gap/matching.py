"""Fuzzy matching utilities for school name matching."""
import re

from thefuzz import fuzz


def standardize_school_name(name: str) -> str:
    """
    Standardize school name for matching.
    
    Transformations:
    - Convert to uppercase
    - Standardize abbreviations (P.S. -> PS, M.S. -> MS, etc.)
    - Remove extra whitespace
    - Remove punctuation except hyphens
    """
    if not name:
        return ""
    
    # Uppercase
    name = name.upper()
    
    # Standardize common abbreviations
    replacements = [
        (r"P\.S\.", "PS"),
        (r"M\.S\.", "MS"),
        (r"H\.S\.", "HS"),
        (r"I\.S\.", "IS"),
        (r"J\.H\.S\.", "JHS"),
        (r"ELEM\.", "ELEMENTARY"),
        (r"SCH\.", "SCHOOL"),
        (r"ACAD\.", "ACADEMY"),
    ]
    
    for pattern, replacement in replacements:
        name = re.sub(pattern, replacement, name)
    
    # Remove punctuation except hyphens
    name = re.sub(r"[^\w\s\-]", "", name)
    
    # Normalize whitespace
    name = " ".join(name.split())
    
    return name


def fuzzy_match_score(name1: str, name2: str) -> int:
    """
    Calculate fuzzy match score between two school names.
    
    Returns score from 0-100 (100 = exact match).
    Uses token_sort_ratio for word-order independence.
    """
    std1 = standardize_school_name(name1)
    std2 = standardize_school_name(name2)
    
    # Use token_sort_ratio for word-order independence
    return fuzz.token_sort_ratio(std1, std2)


def find_best_match(target_name: str, candidates: list[dict], 
                    name_field: str = "school_name",
                    threshold: int = 85) -> tuple[dict | None, int]:
    """
    Find the best matching school from a list of candidates.
    
    Args:
        target_name: The school name to match
        candidates: List of dicts containing candidate schools
        name_field: The field name containing school names in candidates
        threshold: Minimum score to consider a match (0-100)
    
    Returns:
        Tuple of (best_match_dict, score) or (None, 0) if no match above threshold
    """
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        score = fuzzy_match_score(target_name, candidate.get(name_field, ""))
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        return best_match, best_score
    return None, 0

