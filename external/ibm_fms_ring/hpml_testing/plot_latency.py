import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_from_csv(csv_path):
    # Load data
    df = pd.read_csv(csv_path)

    # Sort to ensure proper curve shapes
    df = df.sort_values(by=["weak_pct", "seq_len"])

    # Plot grouped by weak_pct
    plt.figure(figsize=(10, 6))

    for weak_pct, group in df.groupby("weak_pct"):
        plt.plot(group["seq_len"], group["avg_ms"],
                 marker="o", label=f"Weak % = {weak_pct}")

    # Log axes for exponential scaling
    plt.xscale("log")
    plt.yscale("log")

    # Labels & formatting
    plt.xlabel("Sequence Length")
    plt.ylabel("Average Latency (ms)")
    plt.title("Latency vs Sequence Length")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_latency.py <path_to_csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    plot_from_csv(csv_file)
