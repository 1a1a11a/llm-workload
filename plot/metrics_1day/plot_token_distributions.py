#!/usr/bin/env python3

# Used for the one-day trace. 
# This script plots the distribution of input and output tokens for a given model.
# It also plots the distribution of the ratio of output to input tokens.
# It is used to analyze the token distributions of the models and to identify any anomalies.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set large font sizes for better readability
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 24
})

def load_model_data(model_name, requests_dir='requests'):
    """Load input_tokens and output_tokens for a specific model"""
    file_path = Path(requests_dir) / f"{model_name}.csv"
    if not file_path.exists():
        print(f"  -> File not found: {file_path}")
        return None, None

    try:
        df = pd.read_csv(file_path)
        if 'input_tokens' not in df.columns or 'output_tokens' not in df.columns:
            print(f"  -> Missing required columns in {model_name}")
            return None, None
        input_tokens = df['input_tokens'].values
        output_tokens = df['output_tokens'].values
        return input_tokens, output_tokens
    except Exception as e:
        print(f"  -> Error loading {model_name}: {e}")
        return None, None

def plot_cdf(data_dict, title, xlabel, output_file):
    """Plot CDF for multiple datasets"""
    plt.figure(figsize=(12, 8))

    # Define line styles for better differentiation (expanded for 12 models)
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    colors = plt.cm.tab10.colors + plt.cm.Set1.colors  # Combine color palettes for more colors

    for i, (label, data) in enumerate(data_dict.items()):
        if data is not None and len(data) > 0:
            # Sort data for CDF
            sorted_data = np.sort(data)
            # Calculate CDF
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            line_style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]
            plt.plot(sorted_data, y, label=f'{label} (n={len(data):,})',
                    linewidth=3, alpha=0.8, linestyle=line_style, color=color)

    plt.title(title, pad=20)
    plt.xlabel(xlabel)
    plt.xlabel('Token Count')
    plt.ylabel('Cumulative Probability Distribution (CDF)')
    plt.xscale('log')  # Use log scale for token counts
    plt.grid(True, alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def main():
    # Top 10 models by request count + additional requested models
    top_models = [
        # 'DeepSeek-V3-0324',
        # 'DeepSeek-TNG-R1T2-Chimera',
        # # 'Mistral-Small-24B-Instruct-2501',
        # # 'Devstral-Small-2505',
        # 'DeepSeek-R1-0528',
        # # 'Hermes-4-14B',
        # 'DeepSeek-V3.1-Terminus',
        # # 'GLM-4.5-Air',
        # # 'Qwen3-14B',
        # # 'Qwen2.5-VL-32B-Instruct',
        # 'GLM-4.6-FP8',
        # 'Qwen3-Coder-480B-A35B-Instruct-FP8',
        # "GLM-4.6-turbo",
        # 'Qwen3-Coder-30B-A3B-Instruct',
    ]

    # Load data for all models
    input_tokens_data = {}
    output_tokens_data = {}
    input_output_ratios_data = {}

    print("Loading data for top 10 models...")
    for model in top_models:
        print(f"Loading {model}...")
        input_tokens, output_tokens = load_model_data(model)

        if input_tokens is not None and output_tokens is not None:
            input_tokens_data[model] = input_tokens
            output_tokens_data[model] = output_tokens

            # Calculate input/output ratios (avoid division by zero)
            ratios = []
            for i, o in zip(input_tokens, output_tokens):
                if i > 0:  # Avoid division by zero
                    ratios.append(o / i)
            input_output_ratios_data[model] = np.array(ratios)

            print(f"  -> Loaded {len(input_tokens)} samples")
        else:
            print(f"  -> Failed to load {model}")

    print(f"\nLoaded data for {len(input_tokens_data)} models")

    # Create input tokens CDF plot
    print("\nCreating input tokens CDF plot...")
    plot_cdf(
        input_tokens_data,
        'Input Tokens Distribution',
        'Input Tokens',
        'input_tokens_cdf.png'
    )

    # Create output tokens CDF plot
    print("Creating output tokens CDF plot...")
    plot_cdf(
        output_tokens_data,
        'Output Tokens Distribution',
        'Output Tokens',
        'output_tokens_cdf.png'
    )

    # Create input/output ratio CDF plot
    print("Creating input/output ratio CDF plot...")
    plot_cdf(
        input_output_ratios_data,
        'Output/Input Token Distribution',
        'Output/Input Ratio',
        'input_output_ratio_cdf.png'
    )

    print("\nPlotting complete! Generated:")
    print("- input_tokens_cdf.png")
    print("- output_tokens_cdf.png")
    print("- input_output_ratio_cdf.png")

if __name__ == "__main__":
    main()
