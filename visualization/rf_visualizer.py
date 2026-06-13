import matplotlib.pyplot as plt
import os

def plot_granularity_comparison(results, output_path="reports/granularity_comparison.png"):
    """
    Generates and saves a bar chart comparing accuracy across different granularities.

    Args:
        results (dict): Dictionary mapping granularity (int) to accuracy score (float).
        output_path (str): Path where the resulting plot will be saved.
    """
    if not results:
        return
    
    plt.figure(figsize=(10, 6))
    granularities = [str(g) for g in sorted(results.keys())]
    accuracies = [results[int(g)] * 100 for g in granularities]
    
    bars = plt.bar(granularities, accuracies, color='skyblue', edgecolor='navy')
    plt.xlabel('Granularity (Packet Window Size)')
    plt.ylabel('Accuracy (%)')
    plt.title('Network Traffic Classification: Accuracy vs Granularity')
    plt.ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Performance report saved to: {output_path}")