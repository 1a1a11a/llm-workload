import matplotlib.pyplot as plt


def setup_plot_style():
    # Set very large font sizes for better readability
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
            "figure.titlesize": 20,
        }
    )