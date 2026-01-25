"""
Utility functions for data processing and validation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any


def load_data(filepath, **kwargs):
    """
    Load data from various file formats
    
    Args:
        filepath: Path to data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        DataFrame: Loaded data
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, **kwargs)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath, **kwargs)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath, **kwargs)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def validate_team_data(data, required_columns):
    """
    Validate that team data has required columns
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        bool: True if valid, raises ValueError if not
    """
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True


def normalize_team_name(name):
    """
    Normalize team names for consistency
    
    Args:
        name: Raw team name
        
    Returns:
        str: Normalized team name
    """
    # TODO: Implement normalization
    # Handle common variations, abbreviations, etc.
    return name.strip().title()


def get_region_from_seed(seed, num_regions=4):
    """
    Determine region based on seed number
    
    Args:
        seed: Team seed (1-16)
        num_regions: Number of regions (typically 4)
        
    Returns:
        str: Region name
    """
    regions = ['Albany', 'Portland', 'Spokane', 'Greenville']
    # TODO: Implement proper region assignment logic
    return regions[0]  # Placeholder


def format_percentage(value, decimals=1):
    """
    Format decimal as percentage
    
    Args:
        value: Decimal value (0-1)
        decimals: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_ranking(scores, ascending=False):
    """
    Calculate rankings from scores
    
    Args:
        scores: Array of scores
        ascending: If True, lower scores get better ranks
        
    Returns:
        Array: Rankings (1 = best)
    """
    if ascending:
        return pd.Series(scores).rank(ascending=True).astype(int)
    else:
        return pd.Series(scores).rank(ascending=False).astype(int)


def export_predictions(predictions_df, filepath, format='csv'):
    """
    Export predictions to file
    
    Args:
        predictions_df: DataFrame with predictions
        filepath: Output file path
        format: Output format ('csv', 'xlsx', 'json')
    """
    if format == 'csv':
        predictions_df.to_csv(filepath, index=False)
    elif format == 'xlsx':
        predictions_df.to_excel(filepath, index=False)
    elif format == 'json':
        predictions_df.to_json(filepath, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def create_api_response(data, status='success', message='', **kwargs):
    """
    Create standardized API response
    
    Args:
        data: Response data
        status: Status ('success' or 'error')
        message: Optional message
        **kwargs: Additional fields
        
    Returns:
        dict: Formatted API response
    """
    response = {
        'status': status,
        'data': data
    }
    if message:
        response['message'] = message
    response.update(kwargs)
    return response


def handle_error(error, status_code=500):
    """
    Create error response
    
    Args:
        error: Error object or message
        status_code: HTTP status code
        
    Returns:
        tuple: (error_dict, status_code)
    """
    error_message = str(error) if error else 'Unknown error'
    return {
        'status': 'error',
        'message': error_message,
        'code': status_code
    }, status_code
