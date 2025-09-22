import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def plot_trajectory_and_deviation(csv_file, output_file="results/trajectory_and_deviation.png"):
    """
    Plot trajectory and deviations from the saved CSV file.

    :param csv_file: Path to the trajectory CSV file.
    :param output_file: Path to save the plot.
    """
    # Load trajectory data
    trajectory = pd.read_csv(csv_file)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(trajectory["x"], trajectory["y"], label="Vehicle Trajectory", color="red")

    # Add deviation arrows
    for i in range(len(trajectory)):
        if i % 10 == 0:  # Skip every nth point for clarity
            deviation = trajectory.iloc[i]["deviation"]
            x = trajectory.iloc[i]["x"]
            y = trajectory.iloc[i]["y"]
            plt.arrow(x, y, 0, deviation, head_width=0.2, head_length=0.3, fc="green", ec="green", alpha=0.6)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Vehicle Trajectory and Lateral Deviations")
    plt.legend()
    plt.grid()
    # plt.savefig(output_file)
    # print(f"Trajectory plot saved to {output_file}")
    plt.show()

def plot_deviation_heatmap(csv_file, output_file="results/trajectory_deviation_heatmap.png"):
    """
    Plot a heatmap of lateral deviations along the vehicle trajectory.

    :param csv_file: Path to the trajectory CSV file.
    :param output_file: Path to save the heatmap plot.
    """

    # Load trajectory data
    trajectory = pd.read_csv(csv_file)

    # Extract x, y, and deviations
    x = trajectory["x"]
    y = trajectory["y"]
    deviations = trajectory["deviation"]

    # Normalize deviations for color and transparency
    max_deviation = max(abs(deviations))
    norm = Normalize(vmin=-max_deviation, vmax=max_deviation)

    # Create a scatter plot with colors representing deviations
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(x, y, c=deviations, cmap="coolwarm", norm=norm, s=40, alpha=0.8)

    # Add a color bar to indicate deviation magnitudes
    cbar = plt.colorbar(scatter)
    cbar.set_label("Lateral Deviation (m)", rotation=270, labelpad=15)

    # Plot labels and title
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Lateral Deviation Heatmap")
    plt.grid()

    # Save and display the heatmap
    # plt.savefig(output_file)
    # print(f"Heatmap saved to {output_file}")
    plt.show()

def analyze_and_visualize_statistics(csv_file, output_file="results/deviation_statistics.png"):
    """
    Perform statistical analysis of lateral deviations and visualize the results.

    :param csv_file: Path to the trajectory CSV file.
    :param output_file: Path to save the statistical visualization.
    """

    # Load trajectory data
    trajectory = pd.read_csv(csv_file)
    deviations = trajectory["deviation"]

    # Calculate statistics
    mean_deviation = deviations.mean()
    median_deviation = deviations.median()
    max_deviation = deviations.max()
    min_deviation = deviations.min()
    std_deviation = deviations.std()

    # Print statistical summary
    print(f"Statistical Summary:")
    print(f"Mean Deviation: {mean_deviation:.2f} m")
    print(f"Median Deviation: {median_deviation:.2f} m")
    print(f"Max Deviation: {max_deviation:.2f} m")
    print(f"Min Deviation: {min_deviation:.2f} m")
    print(f"Standard Deviation: {std_deviation:.2f} m")

    # Plot histogram of deviations (normalized to percentage)
    plt.figure(figsize=(10, 6))
    plt.hist(
        deviations, bins=30, color="skyblue", edgecolor="k", alpha=0.7, weights=np.ones(len(deviations)) * 100 / len(deviations)
    )
    plt.axvline(mean_deviation, color="red", linestyle="--", label=f"Mean = {mean_deviation:.2f} m")
    plt.axvline(median_deviation, color="green", linestyle="--", label=f"Median = {median_deviation:.2f} m")

    # Adjust the y-axis to percentage scale
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.xlabel("Lateral Deviation (m)")
    plt.ylabel("Percentage")
    plt.title("Histogram of Lateral Deviations")
    plt.legend()

    # Save and display the plot
    # plt.savefig(output_file)
    # print(f"Visualization saved to {output_file}")
    plt.show()

    # Additional statistical insights
    within_10cm = np.sum(np.abs(deviations) <= 0.1) / len(deviations) * 100
    within_20cm = np.sum(np.abs(deviations) <= 0.2) / len(deviations) * 100
    print(f"Percentage of deviations within ±10 cm: {within_10cm:.2f}%")
    print(f"Percentage of deviations within ±20 cm: {within_20cm:.2f}%")

def plot_deviation_vs_distance(csv_file, output_file="results/deviation_vs_distance.png"):
    """
    Plot lateral deviations against the distance traveled from the start.

    :param csv_file: Path to the trajectory CSV file.
    :param output_file: Path to save the visualization.
    """

    # Load trajectory data
    trajectory = pd.read_csv(csv_file)

    # Extract x, y, and deviations
    x = trajectory["x"].to_numpy()
    y = trajectory["y"].to_numpy()
    deviations = trajectory["deviation"].to_numpy()

    # Calculate cumulative distance traveled from the start
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    distances = np.insert(distances, 0, 0)  # Add zero for the starting point
    cumulative_distance = np.cumsum(distances)

    # Normalize deviations for color and transparency
    max_deviation = max(abs(deviations))
    norm = Normalize(vmin=-max_deviation, vmax=max_deviation)
    alpha = np.clip(abs(deviations) / max_deviation, 0.2, 1.0)  # Symmetric alpha transparency

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(deviations, cumulative_distance, c=deviations, cmap="coolwarm", norm=norm, alpha=alpha,
                          s=20, edgecolors='gray', linewidths=0.5)

    # Add a color bar to represent deviation magnitudes
    cbar = plt.colorbar(scatter)
    cbar.set_label("Deviation Magnitude (m)", rotation=270, labelpad=15)

    # Add labels, grid, and title
    plt.axvline(0, color="black", linestyle="--", linewidth=1)  # Zero deviation line
    plt.xlabel("Lateral Deviation (m)")
    plt.ylabel("Distance Traveled (m)")
    plt.title("Lateral Deviation vs. Distance Traveled")
    plt.grid()

    # Save and display the plot
    # plt.savefig(output_file)
    # print(f"Visualization saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    csv_file = 'results/town03/legacy_Town03_ClearNoon_40_None_None_vehicle_trajectory_1.csv'
    # plot_trajectory_and_deviation(csv_file)
    plot_deviation_heatmap(csv_file)
    analyze_and_visualize_statistics(csv_file)
    plot_deviation_vs_distance(csv_file)
