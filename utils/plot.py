import matplotlib.pyplot as plt


def setup_plot_style():
    # Set large font sizes for better readability
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "figure.titlesize": 24,
        }
    )

