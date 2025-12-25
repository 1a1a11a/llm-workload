#!/usr/bin/env python3
"""
Configuration management for the LLM Workload Analysis Toolkit.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class AnalysisConfig:
    """
    Configuration class for managing analysis settings.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration with optional config file.
        
        Args:
            config_file (str, optional): Path to configuration file
        """
        self._config = self._load_default_config()
        
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
            
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration values.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "data": {
                "default_input_file": "/scratch/juncheng/data/prefix_cache/metrics_30day_head.csv",
                "data_directory": "/scratch/juncheng/data/prefix_cache/",
                "per_model_directory": "/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/"
            },
            "output": {
                "base_output_dir": "figures",
                "plot_dpi": 300,
                "plot_bbox_inches": "tight"
            },
            "analysis": {
                "large_dataset_threshold": 1000000,
                "outlier_percentile": 99.9,
                "token_bucket_size": 1000
            },
            "plotting": {
                "default_figsize": (12, 8),
                "boxplot_figsize": (16, 10)
            }
        }
        
    def _load_from_file(self, config_file: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_file (str): Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._config = self._merge_configs(self._config, file_config)
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
            
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base (Dict[str, Any]): Base configuration
            override (Dict[str, Any]): Override configuration
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path.
        
        Args:
            key_path (str): Dot-separated path to configuration value
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key_path.split('.')
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation path.
        
        Args:
            key_path (str): Dot-separated path to configuration value
            value (Any): Value to set
        """
        keys = key_path.split('.')
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        
    def save(self, config_file: str):
        """
        Save current configuration to JSON file.
        
        Args:
            config_file (str): Path to save configuration file
        """
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(self._config, f, indent=2)
            
    def get_data_directory(self) -> str:
        """Get the base data directory."""
        return self.get('data.data_directory')
        
    def get_per_model_directory(self) -> str:
        """Get the per-model data directory."""
        return self.get('data.per_model_directory')
        
    def get_output_directory(self) -> str:
        """Get the base output directory."""
        return self.get('output.base_output_dir')
        
    def get_large_dataset_threshold(self) -> int:
        """Get the threshold for large datasets."""
        return self.get('analysis.large_dataset_threshold')
        
    def get_outlier_percentile(self) -> float:
        """Get the outlier percentile threshold."""
        return self.get('analysis.outlier_percentile')
        
    def get_token_bucket_size(self) -> int:
        """Get the token bucket size for box plots."""
        return self.get('analysis.token_bucket_size')
        
    def get_default_figsize(self) -> tuple:
        """Get the default figure size."""
        return self.get('plotting.default_figsize')
        
    def get_boxplot_figsize(self) -> tuple:
        """Get the boxplot figure size."""
        return self.get('plotting.boxplot_figsize')
</environment_details>