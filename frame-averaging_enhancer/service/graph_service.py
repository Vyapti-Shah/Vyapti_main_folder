"""Service responsible for visualizing and exporting sharpness data."""

import csv
import logging

import matplotlib.pyplot as plt

from models import FrameScore

logger = logging.getLogger(__name__)


class GraphService:
    """Handles visualization and CSV export of sharpness data."""

    def save_graph(
        self,
        scores: list[FrameScore],
        best_score: FrameScore,
        output_path: str,
        scene_number: int,
    ) -> None:
        """Plots sharpness over time and saves the figure to disk."""
        frame_indices = [score.frame_number for score in scores]
        sharpness_values = [score.sharpness for score in scores]

        plt.figure(figsize=(10, 5))
        plt.plot(
            frame_indices,
            sharpness_values,
            marker="",
            color="b",
            label="Sharpness (Laplacian Variance)",
        )
        plt.axvline(
            x=best_score.frame_number,
            color="r",
            linestyle="--",
            label=f"Best Frame ({best_score.frame_number})",
        )
        plt.title(f"Scene {scene_number} Sharpness Over Time")
        plt.xlabel("Frame Number")
        plt.ylabel("Sharpness Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

        logger.info("Saved sharpness graph to: %s", output_path)

    def save_csv(self, scores: list[FrameScore], output_path: str) -> None:
        """Writes frame sharpness scores to a CSV file."""
        with open(output_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Frame Number", "Sharpness Score"])
            for score in scores:
                writer.writerow([score.frame_number, f"{score.sharpness:.2f}"])

        logger.info("Saved sharpness data to: %s", output_path)
