#!/usr/bin/env python3
"""
Script to plot TTFT vs Input Token Length, grouped by model.
This script analyzes the relationship between input token length and time to first token (TTFT)
for different models in the LLM workload metrics data.

The script creates a scatter plot showing how TTFT varies with input token length,
with different models represented by different colors.

For large datasets (>1M requests), it creates box plots using 1k token buckets
to better visualize the distribution of TTFT across different input token length ranges.

Usage:
    python3 plot_per_model_ttft_vs_tokens.py [input_file]

If no input file is provided, it defaults to loading from:
    /scratch/juncheng/data/prefix_cache/metrics_30day_head.csv

Requirements:
    - pandas
    - matplotlib
    - numpy
    - pyarrow (for parquet support)
    - psutil (for memory management)

Output:
    - A scatter plot PNG file saved to figures/per_model_ttft_analysis/{folder_name}/
    - For large datasets (>1M requests): box plot with 1k token buckets
    - Statistics printed to console
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.base_analysis import BaseAnalysis
from analysis.config import AnalysisConfig


class TTFTAnalysis(BaseAnalysis):
    """
    Analysis class for TTFT vs Input Token Length analysis.
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize the TTFT analysis.
        
        Args:
            output_dir (str): Base directory for saving outputs
        """
        super().__init__(output_dir)
        self.config = AnalysisConfig()
        
    def plot_ttft_boxplot_by_tokens(self, df, output_dir="figures", file_path=None):
        """
        Create box plots for TTFT vs Input Token Length using 1k token buckets.
        This function is used for large datasets (>1M requests) to better visualize
        the distribution of TTFT across different input token length ranges.
        
        Args:
            df (pandas.DataFrame): The metrics data DataFrame containing columns:
                - input_tokens: Number of tokens in the input
                - ttft: Time to first token (seconds)
                - model_name: (Optional) Model identifier for coloring
            output_dir (str): Directory to save the plot. Defaults to "figures"
            file_path (str): Path to the input file, used for generating filename
            
        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter data to only include rows with both ttft and input_tokens
        valid_data = self.filter_valid_data(df, ['ttft', 'input_tokens'])
        
        if len(valid_data) == 0:
            print("No valid data with both ttft and input_tokens available")
            return
        
        # Remove outliers on TTFT to focus on main distribution
        y_threshold = np.percentile(valid_data['ttft'], self.config.get_outlier_percentile())
        valid_data = valid_data[valid_data['ttft'] <= y_threshold]
        
        # Create token buckets based on configuration
        bucket_size = self.config.get_token_bucket_size()
        valid_data = valid_data.copy()
        valid_data['token_bucket'] = (valid_data['input_tokens'] // bucket_size) * bucket_size
        
        # Get bucket ranges and statistics
        bucket_stats_raw = valid_data.groupby('token_bucket')['ttft'].agg(['count', 'mean', 'median', 'std']).reset_index()
        
        # Prepare data for box plot with merging logic
        bucket_data = []
        bucket_labels = []
        final_bucket_stats = []
        
        current_bucket_ttfts = []
        current_bucket_start = None
        current_count = 0
        
        # Sort buckets to process sequentially
        sorted_buckets = bucket_stats_raw.sort_values('token_bucket')
        
        for _, row in sorted_buckets.iterrows():
            bucket_start = row['token_bucket']
            if current_bucket_start is None:
                current_bucket_start = bucket_start
                
            bucket_ttft = valid_data[valid_data['token_bucket'] == bucket_start]['ttft'].values
            current_bucket_ttfts.extend(bucket_ttft)
            current_count += len(bucket_ttft)
            
            # If current accumulation has enough requests, OR it's the last bucket
            if current_count >= 100:
                bucket_end = bucket_start + bucket_size
                label = f'{int(current_bucket_start//bucket_size)}K-{int(bucket_end//bucket_size)}K'
                data = np.array(current_bucket_ttfts)
                
                bucket_data.append(data)
                bucket_labels.append(label)
                final_bucket_stats.append({
                    'label': label,
                    'count': current_count,
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'std': np.std(data)
                })
                
                # Reset for next merged bucket
                current_bucket_ttfts = []
                current_bucket_start = None
                current_count = 0
                
        # Handle leftover data if it didn't reach 100 but we're at the end
        if current_count > 0:
            if bucket_data:
                # Merge into last bucket
                bucket_data[-1] = np.concatenate([bucket_data[-1], current_bucket_ttfts])
                # Update last bucket label
                old_start = bucket_labels[-1].split('-')[0]
                new_end = f'{int((sorted_buckets.iloc[-1]["token_bucket"] + bucket_size)//bucket_size)}K'
                new_label = f'{old_start}-{new_end}'
                bucket_labels[-1] = new_label
                # Update stats
                final_bucket_stats[-1]['label'] = new_label
                final_bucket_stats[-1]['count'] += current_count
                final_bucket_stats[-1]['mean'] = np.mean(bucket_data[-1])
                final_bucket_stats[-1]['median'] = np.median(bucket_data[-1])
                final_bucket_stats[-1]['std'] = np.std(bucket_data[-1])
            else:
                # Only one bucket total, less than 100 requests
                bucket_end = sorted_buckets.iloc[-1]['token_bucket'] + bucket_size
                label = f'{int(current_bucket_start//bucket_size)}K-{int(bucket_end//bucket_size)}K'
                data = np.array(current_bucket_ttfts)
                bucket_data.append(data)
                bucket_labels.append(label)
                final_bucket_stats.append({
                    'label': label,
                    'count': current_count,
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'std': np.std(data)
                })

        bucket_stats = pd.DataFrame(final_bucket_stats)
        
        if len(bucket_data) == 0:
            print("No buckets with data found")
            return
        
        # Create the box plot
        plt.figure(figsize=self.config.get_boxplot_figsize())
        
        # Create box plot
        box_plot = plt.boxplot(bucket_data, 
                              tick_labels=None, # Set labels manually for better density control
                              patch_artist=True,
                              showfliers=False,
                              whis=[10, 90],
                              showmeans=True,
                              meanprops={"marker": "v", "markeredgecolor": "black"})
        
        # Color the boxes
        colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(bucket_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.xlabel('Input Token Length Buckets')
        plt.ylabel('Time To First Token (seconds)')
        total_requests = len(valid_data)
        
        # Generate trace name from file path
        trace_name = "Unknown Trace"
        if file_path:
            trace_name = os.path.splitext(os.path.basename(file_path))[0]
            
        plt.title(f'TTFT Distribution by Input Token Length Buckets\nTrace: {trace_name} (N={total_requests:,})')
        
        # Reduce x-tick label density
        ax = plt.gca()
        # Aim for ~20 labels max
        n = max(1, len(bucket_labels) // 20)
        tick_indices = np.arange(1, len(bucket_labels) + 1, n)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([bucket_labels[i-1] for i in tick_indices], rotation=90)
        
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Generate filename based on input file name
        if file_path:
            input_filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(input_filename)[0]
            save_filename = f"{name_without_ext}_ttft_boxplot_vs_tokens.png"
        else:
            save_filename = "ttft_boxplot_vs_input_tokens.png"
        
        self.save_plot(plt.gcf(), save_filename, output_dir)
        
        # Save plot data
        data_dir = os.path.join(output_dir, "plot_data")
        os.makedirs(data_dir, exist_ok=True)
        data_filename = save_filename.replace(".png", ".csv")
        bucket_stats.to_csv(os.path.join(data_dir, data_filename), index=False)
        
        # Print bucket statistics
        print("\nTTFT Bucket Statistics:")
        print(f"  Total data points: {total_requests:,}")
        print(f"  Number of buckets: {len(bucket_data)}")
        print(f"  Bucket ranges: {bucket_labels[0]} to {bucket_labels[-1]}")
        print("\nBucket Details:")
        for _, row in bucket_stats.iterrows():
            print(f"  {row['label']} tokens: {row['count']:,} requests, "
                  f"median TTFT: {row['median']:.3f}s, mean TTFT: {row['mean']:.3f}s")

    def plot_ttft_vs_input_tokens(self, df, output_dir="figures", file_path=None):
        """
        Plot TTFT vs Input Token Length, with separate colors for each model.
        
        This function creates a scatter plot showing the relationship between
        input token length and time to first token (TTFT) for different models.
        
        The plot includes:
        - Scatter points for each data point, colored by model
        - Appropriate labels and title
        
        Args:
            df (pandas.DataFrame): The metrics data DataFrame containing columns:
                - input_tokens: Number of tokens in the input
                - ttft: Time to first token (seconds)
                - model_name: (Optional) Model identifier for coloring
            output_dir (str): Directory to save the plot. Defaults to "figures"
            file_path (str): Path to the input file, used for generating filename
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> plot_ttft_vs_input_tokens(df, "output/plots", "/path/to/data.csv")
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter data to only include rows with both ttft and input_tokens
        valid_data = self.filter_valid_data(df, ['ttft', 'input_tokens'])
        
        if len(valid_data) == 0:
            print("No valid data with both ttft and input_tokens available")
            return
        
        # Remove outliers on both axes
        valid_data = self.remove_outliers(valid_data, 'input_tokens', self.config.get_outlier_percentile())
        valid_data = self.remove_outliers(valid_data, 'ttft', self.config.get_outlier_percentile())
        
        # Create the scatter plot
        plt.figure(figsize=self.config.get_default_figsize())
        
        # Plot with different colors for different models if available
        if 'model_name' in df.columns:
            # Get unique models
            models = valid_data['model_name'].unique()
            colors = plt.get_cmap('Set1')(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                model_data = valid_data[valid_data['model_name'] == model]
                plt.scatter(model_data['input_tokens'], model_data['ttft'], 
                           alpha=0.7, label=str(model), color=colors[i])
            
            plt.legend(loc='best')
        else:
            plt.scatter(valid_data['input_tokens'], valid_data['ttft'], alpha=0.7)
        
        plt.xlabel('Input Token Length')
        plt.ylabel('TTFT (Time To First Token) [seconds]')
        
        # Generate trace name from file path
        trace_name = "Unknown Trace"
        if file_path:
            trace_name = os.path.splitext(os.path.basename(file_path))[0]
            
        plt.title(f'TTFT vs Input Token Length\nTrace: {trace_name} (N={len(valid_data):,})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Generate filename based on input file name
        if file_path:
            input_filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(input_filename)[0]
            save_filename = f"{name_without_ext}_ttft_vs_tokens.png"
        else:
            save_filename = "ttft_vs_input_tokens.png"
        
        self.save_plot(plt.gcf(), save_filename, output_dir)
        
        # Save plot data
        data_dir = os.path.join(output_dir, "plot_data")
        os.makedirs(data_dir, exist_ok=True)
        data_filename = save_filename.replace(".png", ".csv")
        cols_to_save = ['input_tokens', 'ttft', 'model_name'] if 'model_name' in valid_data.columns else ['input_tokens', 'ttft']
        valid_data[cols_to_save].to_csv(os.path.join(data_dir, data_filename), index=False)
        
        # Print basic statistics
        print("\nTTFT vs Input Tokens Statistics:")
        print(f"  Data points: {len(valid_data)}")
        print(f"  Input tokens - Mean: {valid_data['input_tokens'].mean():.2f}, Std: {valid_data['input_tokens'].std():.2f}")

    def run_analysis(self, file_path: str):
        """
        Run the TTFT vs Input Tokens analysis on the given file.
        
        Args:
            file_path (str): Path to the input file
        """
        # Create output directory with proper structure
        output_dir = self.get_output_directory(file_path, "per_model_ttft_analysis")
        
        df = self.load_data(file_path, apply_transforms=True)
        
        # Filter and keep only necessary columns
        valid_cols = [c for c in ['input_tokens', 'ttft', 'model_name'] if c in df.columns]
        df = self.filter_valid_data(df, ['ttft', 'input_tokens'])
        df = df[valid_cols]
        
        if len(df) < 1000:
            print(f"Skipping trace with only {len(df)} requests (minimum 1000 required).")
            return
        
        # Choose plotting method based on dataset size
        if len(df) > self.config.get_large_dataset_threshold():  # Use box plot for large datasets
            print(f"Large dataset detected ({len(df):,} requests). Using box plot with token buckets.")
            self.plot_ttft_boxplot_by_tokens(df, output_dir, file_path)
        else:  # Use scatter plot for smaller datasets
            print(f"Using scatter plot for {len(df):,} requests.")
            self.plot_ttft_vs_input_tokens(df, output_dir, file_path)
            
    def main(self):
        """
        Main function to load data and generate the TTFT vs Input Tokens plot.
        """
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            file_path = self.config.get('data.default_input_file', 
                                      "/scratch/juncheng/data/prefix_cache/metrics_30day_head.csv")
            
        self.run_analysis(file_path)


if __name__ == "__main__":
    analysis = TTFTAnalysis()
    analysis.main()