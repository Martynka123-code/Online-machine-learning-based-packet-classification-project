import pickle
import matplotlib.pyplot as plt


def analyze_hyperopt_results():
    try:
        with open("hyperopt_trials.pkl", "rb") as f:
            trials = pickle.load(f)
    except FileNotFoundError:
        print("[!] File hyperopt_trials.pkl not found. Run optimize.py first!")
        return

    # Filter only successful trials
    successful_trials = [t for t in trials.trials if t['result']['status'] == 'ok']

    if not successful_trials:
        print("[!] No successful experiments to display.")
        return

    losses = [t['result']['loss'] for t in successful_trials]
    lrs = [t['misc']['vals']['learning_rate'][0] for t in successful_trials]
    dropouts = [t['misc']['vals']['dropout'][0] for t in successful_trials]

    # Define option lists exactly as defined in the 'space' dictionary
    batch_size_options = [16, 32, 64, 128]
    kernel_size_options = [2, 3, 5, 7]
    filter_options = ["32->16", "48->24", "64->32", "80->40"]

    batch_sizes = [batch_size_options[t['misc']['vals']['batch_size'][0]] for t in successful_trials]
    kernel_sizes = [kernel_size_options[t['misc']['vals']['kernel_size'][0]] for t in successful_trials]
    filters_idx = [filter_options[t['misc']['vals']['conv_filters'][0]] for t in successful_trials]

    # Create a grid of plots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Hyperparameter Impact on Validation Loss (Lower Loss is better)', fontsize=16,
                 fontweight='bold')

    # Plot 1: Learning Rate
    axs[0, 0].scatter(lrs, losses, color='blue', s=100, edgecolors='black', alpha=0.7)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel('Learning Rate')
    axs[0, 0].set_ylabel('Validation Loss')
    axs[0, 0].set_title('Learning Rate vs Loss')
    axs[0, 0].grid(True, which="both", ls="--")

    # Plot 2: Dropout
    axs[0, 1].scatter(dropouts, losses, color='green', s=100, edgecolors='black', alpha=0.7)
    axs[0, 1].set_xlabel('Dropout Rate')
    axs[0, 1].set_ylabel('Validation Loss')
    axs[0, 1].set_title('Dropout vs Loss')
    axs[0, 1].grid(True, ls="--")

    # Plot 3: Batch Size
    axs[0, 2].scatter(batch_sizes, losses, color='red', s=100, edgecolors='black', alpha=0.7)
    axs[0, 2].set_xlabel('Batch Size')
    axs[0, 2].set_ylabel('Validation Loss')
    axs[0, 2].set_title('Batch Size vs Loss')
    axs[0, 2].set_xticks(batch_size_options)
    axs[0, 2].grid(True, ls="--")

    # Plot 4: Kernel Size
    axs[1, 0].scatter(kernel_sizes, losses, color='purple', s=100, edgecolors='black', alpha=0.7)
    axs[1, 0].set_xlabel('Kernel Size')
    axs[1, 0].set_ylabel('Validation Loss')
    axs[1, 0].set_title('Kernel Size vs Loss')
    axs[1, 0].set_xticks(kernel_size_options)
    axs[1, 0].grid(True, ls="--")

    # Plot 5: Number of filters (Architecture)
    axs[1, 1].scatter(filters_idx, losses, color='orange', s=100, edgecolors='black', alpha=0.7)
    axs[1, 1].set_xlabel('Filter Configuration (conv1 -> conv2)')
    axs[1, 1].set_ylabel('Validation Loss')
    axs[1, 1].set_title('Number of Filters vs Loss')
    axs[1, 1].grid(True, ls="--")

    # Leave the last subplot empty or remove it
    fig.delaxes(axs[1, 2])

    plt.tight_layout()
    plt.savefig("optimization_report.png", dpi=300)
    print("[+] Optimization analysis plot generated and saved to: optimization_report.png")
    plt.show()


if __name__ == "__main__":
    analyze_hyperopt_results()