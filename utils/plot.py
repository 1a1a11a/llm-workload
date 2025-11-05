import matplotlib.pyplot as plt


def setup_plot_style():
    # Set large font sizes for better readability
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.titlesize": 24,
        }
    )
