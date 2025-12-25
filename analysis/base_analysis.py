#!/usr/bin/env python3
"""
Base analysis class for LLM workload metrics analysis.
This provides common functionality and structure for all analysis scripts.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Add the parent directory to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from readers.data_loader import load_metrics_dataframe
from utils.plot import setup_plot_style

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseAnalysis:
    """
    Base class for all analysis scripts.
    Provides common functionality like data loading, output directory management,
    and plotting utilities.
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize the base analysis class.
        
        Args:
            output_dir (str): Base directory for saving outputs
        """
        self.output_dir = output_dir
        self.setup_plot_style()
        
    def setup_plot_style(self):
        """Set up consistent plot styling."""
        setup_plot_style()
        
    def load_data(self, file_path: str, apply_transforms: bool = True) -> pd.DataFrame:
        """
        Load and process metrics data.
        
        Args:
            file_path (str): Path to the data file
            apply_transforms (bool): Whether to apply data transformations
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            df = load_metrics_dataframe(file_path, apply_transforms=apply_transforms)
            logger.info(f"Loaded data from {file_path}: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
            
    def get_output_directory(self, file_path: str, subdirectory: str = "") -> str:
        """
        Get the appropriate output directory for a given input file.
        
        Args:
            file_path (str): Path to the input file
            subdirectory (str): Subdirectory name for organizing outputs
            
        Returns:
            str: Output directory path
        """
        path_obj = Path(file_path)
        
        # Extract folder name from file path for proper output organization
        # Expected path structure: .../per_model/{folder_name}/{model_name}.csv
        folder_name = "unknown"
        if "per_model" in path_obj.parts:
            try:
                idx = path_obj.parts.index("per_model")
                if len(path_obj.parts) > idx + 1:
                    folder_name = path_obj.parts[idx + 1]
            except ValueError:
                pass
                
        # Create base output directory
        output_path = Path(self.output_dir) / subdirectory / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        return str(output_path)
        
    def save_plot(self, figure: plt.Figure, filename: str, output_dir: str, 
                  dpi: int = 300, bbox_inches: str = "tight") -> str:
        """
        Save a matplotlib figure to file.
        
        Args:
            figure (plt.Figure): Matplotlib figure to save
            filename (str): Name of the file to save
            output_dir (str): Directory to save the file
            dpi (int): DPI for the saved image
            bbox_inches (str): Bbox_inches parameter for savefig
            
        Returns:
            str: Full path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
        figure.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(figure)
        logger.info(f"Saved plot to {full_path}")
        return full_path
        
    def filter_valid_data(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """
        Filter DataFrame to only include rows with valid data for required columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            required_columns (List[str]): List of required column names
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        valid_data = df.dropna(subset=required_columns)
        logger.info(f"Filtered data: {len(df)} -> {len(valid_data)} rows")
        return valid_data
        
    def remove_outliers(self, df: pd.DataFrame, column: str, percentile: float = 99.9) -> pd.DataFrame:
        """
        Remove outliers from a column based on percentile.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column name to remove outliers from
            percentile (float): Percentile threshold (default 99.9)
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        if column not in df.columns:
            return df
            
        threshold = df[column].quantile(percentile / 100)
        filtered_df = df[df[column] <= threshold]
        logger.info(f"Removed outliers from {column}: {len(df)} -> {len(filtered_df)} rows")
        return filtered_df