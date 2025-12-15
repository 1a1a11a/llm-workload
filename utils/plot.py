import matplotlib.pyplot as plt


def setup_plot_style():
    # Set very large font sizes for better readability
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 32,
            "axes.titlesize": 36,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 28,
            "figure.titlesize": 40,
        }
    )