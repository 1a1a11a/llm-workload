import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
from pathlib import Path

# Add readers module to path for data loading utilities
sys.path.append(str(Path(__file__).parent.parent))
from readers.data_loader import load_metrics_dataframe
from utils.plot import setup_plot_style


def load_chutes_models_map():
    """Load the chutes to models mapping from the JSON file."""
    try:
        # Try to load from data directory first, then root directory
        possible_paths = [
            "data/chutes_models.json",
            "chutes_models.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Create mapping from chute_id to model name
                chute_to_model = {}
                if 'items' in data:
                    for item in data['items']:
                        chute_id = item.get('chute_id')
                        model_name = item.get('name', 'Unknown Model')
                        if chute_id:
                            chute_to_model[chute_id] = model_name
                
                return chute_to_model
                
        print("Warning: chutes_models.json not found, using chute_id as model names")
        return {}
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load chutes_models.json: {e}")
        print("Using chute_id as model names")
        return {}


def validate_data_columns(df, required_columns):
    """Validate that the dataframe has all required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for alternative column names and provide helpful suggestions
    alternative_mappings = {
        'user_id': ['user', 'userid', 'user_id'],
        'chute_id': ['chute_id', 'model_id', 'chute', 'model']
    }
    
    for required, alternatives in alternative_mappings.items():
        if required not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    print(f"Info: Found '{alt}' column, did you mean to use '{required}' instead?")
                    break


def plot_requests_per_user_cdf(df, output_dir):
    """
    Plots a CDF of the number of requests each user sends.
    """
    user_requests = df.groupby("user_id").size()

    sorted_requests = np.sort(user_requests)
    yvals = np.arange(len(sorted_requests)) / float(len(sorted_requests) - 1)

    # Dump data for replotting
    dump_path = os.path.join(output_dir, "requests_per_user_cdf.csv")
    pd.DataFrame({
        "requests": sorted_requests,
        "cdf": yvals
    }).to_csv(dump_path, index=False)
    print(f"Data dumped to {dump_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_requests, yvals)
    plt.xlabel("Number of Requests")
    plt.ylabel("Fraction of Users")
    plt.title("CDF of Requests per User")
    plt.grid(True)
    plt.xscale("log")

    output_path = os.path.join(output_dir, "requests_per_user_cdf.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_models_per_user_cdf(df, chute_to_model_map, output_dir):
    """
    Plots a CDF of the number of models each user accesses.
    """
    # Group by user_id and chute_id, then map chute_id to model names
    user_chutes = df.groupby("user_id")["chute_id"].apply(set)
    
    # Map chute_ids to model names
    user_models = user_chutes.apply(lambda chutes: len([chute_to_model_map.get(chute, chute) for chute in chutes]))

    sorted_models = np.sort(user_models)
    yvals = np.arange(len(sorted_models)) / float(len(sorted_models) - 1)

    # Dump data for replotting
    dump_path = os.path.join(output_dir, "models_per_user_cdf.csv")
    pd.DataFrame({
        "models": sorted_models,
        "cdf": yvals
    }).to_csv(dump_path, index=False)
    print(f"Data dumped to {dump_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_models, yvals)
    plt.xlabel("Number of Models")
    plt.ylabel("Fraction of Users")
    plt.title("CDF of Models Accessed per User")
    plt.grid(True)
    plt.xscale("log")

    output_path = os.path.join(output_dir, "models_per_user_cdf.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    # Set up plot style
    setup_plot_style()
    
    # Default input file if not provided as a command-line argument
    default_input_file = "metrics_30day.csv"
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input_file

    output_dir = "figures/user_analysis/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load chutes to models mapping
    chute_to_model_map = load_chutes_models_map()
    
    try:
        # Try to use standardized data loader first, fallback to direct CSV loading
        try:
            df = load_metrics_dataframe(input_file)
        except ImportError:
            print("Info: Standard data loader not available, using direct CSV loading")
            df = pd.read_csv(input_file)
            
    except FileNotFoundError:
        print(
            f"Error: Input file '{input_file}' not found. Please provide a valid path."
        )
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Validate required columns
    required_columns = ["user_id", "chute_id"]
    try:
        validate_data_columns(df, required_columns)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique models/chutes: {df['chute_id'].nunique()}")

    plot_requests_per_user_cdf(df, output_dir)

    plot_models_per_user_cdf(df, chute_to_model_map, output_dir)


if __name__ == "__main__":
    main()
