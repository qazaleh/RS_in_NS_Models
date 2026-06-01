import csv
import json
import os
from collections import Counter

import matplotlib.pyplot as plt


class ResultVisualizer:
    def __init__(
        self,
        metrics_path="../results/metrics/uncertainty_filter_results.json",
        shortcut_csv_path="../results/metrics/shortcut_analysis.csv",
        figures_dir="../results/figures",
    ):
        self.metrics_path = metrics_path
        self.shortcut_csv_path = shortcut_csv_path
        self.figures_dir = figures_dir

        os.makedirs(self.figures_dir, exist_ok=True)

        self.metrics = self._load_metrics()
        self.shortcut_rows = self._load_shortcut_rows()

    def _load_metrics(self):
        with open(self.metrics_path, "r") as f:
            return json.load(f)

    def _load_shortcut_rows(self):
        with open(self.shortcut_csv_path, "r") as f:
            return list(csv.DictReader(f))

    def _get_baseline_and_filter_metrics(self):
        baseline = self.metrics["baseline_cnn"]

        filter_key = None
        for key in self.metrics.keys():
            if key != "baseline_cnn" and key != "filter_change_analysis":
                filter_key = key
                break

        if filter_key is None:
            raise ValueError("No filter metrics found in metrics JSON.")

        filtered = self.metrics[filter_key]

        return baseline, filtered, filter_key

    def plot_accuracy_comparison(self):
        baseline, filtered, _ = self._get_baseline_and_filter_metrics()

        labels = ["Reasoning Accuracy", "Concept Accuracy"]
        baseline_values = [
            baseline["reasoning_accuracy"],
            baseline["concept_accuracy"],
        ]
        filtered_values = [
            filtered["reasoning_accuracy"],
            filtered["concept_accuracy"],
        ]

        x = range(len(labels))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(
            [i - width / 2 for i in x], baseline_values, width, label="Baseline CNN"
        )
        plt.bar(
            [i + width / 2 for i in x], filtered_values, width, label="Filtered Model"
        )

        plt.xticks(list(x), labels)
        plt.ylim(0.95, 1.0)
        plt.ylabel("Accuracy")
        plt.title("Baseline vs Filtered Accuracy")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.figures_dir, "accuracy_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved {save_path}")

    def plot_shortcut_count_comparison(self):
        baseline, filtered, _ = self._get_baseline_and_filter_metrics()

        labels = ["Baseline CNN", "Filtered Model"]
        values = [
            baseline["shortcut_count"],
            filtered["shortcut_count"],
        ]

        plt.figure(figsize=(7, 5))
        plt.bar(labels, values)
        plt.ylabel("Shortcut Count")
        plt.title("Reasoning Shortcut Count Comparison")
        plt.tight_layout()

        save_path = os.path.join(self.figures_dir, "shortcut_count_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved {save_path}")

    def plot_shortcut_frequency_comparison(self):
        baseline, filtered, _ = self._get_baseline_and_filter_metrics()

        labels = ["Baseline CNN", "Filtered Model"]
        values = [
            baseline["shortcut_frequency"],
            filtered["shortcut_frequency"],
        ]

        plt.figure(figsize=(7, 5))
        plt.bar(labels, values)
        plt.ylabel("Shortcut Frequency")
        plt.title("Reasoning Shortcut Frequency Comparison")
        plt.tight_layout()

        save_path = os.path.join(self.figures_dir, "shortcut_frequency_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved {save_path}")

    def plot_top_shortcut_confusions(self, top_n=10):
        confusion_counter = Counter()

        for row in self.shortcut_rows:
            true_d1 = int(row["true_digit1"])
            true_d2 = int(row["true_digit2"])

            cnn_d1 = int(row["cnn_pred_digit1"])
            cnn_d2 = int(row["cnn_pred_digit2"])

            if true_d1 != cnn_d1:
                confusion_counter[f"{true_d1}→{cnn_d1}"] += 1

            if true_d2 != cnn_d2:
                confusion_counter[f"{true_d2}→{cnn_d2}"] += 1

        most_common = confusion_counter.most_common(top_n)

        labels = [item[0] for item in most_common]
        values = [item[1] for item in most_common]

        plt.figure(figsize=(9, 5))
        plt.bar(labels, values)
        plt.xlabel("Digit Confusion")
        plt.ylabel("Count")
        plt.title("Top Digit Confusions in Shortcut Cases")
        plt.tight_layout()

        save_path = os.path.join(self.figures_dir, "top_shortcut_confusions.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved {save_path}")

    def plot_filter_probability_shift(self):
        fixed_rows = [
            row
            for row in self.shortcut_rows
            if str(row["filter_fixed_shortcut"]).lower() == "true"
        ]

        if len(fixed_rows) == 0:
            print("No fixed shortcut cases found. Skipping probability shift plot.")
            return

        cnn_digit1_true_probs = []
        cnn_digit2_true_probs = []
        filtered_digit1_true_probs = []
        filtered_digit2_true_probs = []

        for row in fixed_rows:
            cnn_digit1_true_probs.append(float(row["cnn_digit1_true_prob"]))
            cnn_digit2_true_probs.append(float(row["cnn_digit2_true_prob"]))

            filtered_digit1_true_probs.append(float(row["filtered_digit1_true_prob"]))
            filtered_digit2_true_probs.append(float(row["filtered_digit2_true_prob"]))

        avg_cnn_digit1 = sum(cnn_digit1_true_probs) / len(cnn_digit1_true_probs)
        avg_cnn_digit2 = sum(cnn_digit2_true_probs) / len(cnn_digit2_true_probs)

        avg_filtered_digit1 = sum(filtered_digit1_true_probs) / len(
            filtered_digit1_true_probs
        )
        avg_filtered_digit2 = sum(filtered_digit2_true_probs) / len(
            filtered_digit2_true_probs
        )

        labels = [
            "CNN Digit1",
            "Filtered Digit1",
            "CNN Digit2",
            "Filtered Digit2",
        ]

        values = [
            avg_cnn_digit1,
            avg_filtered_digit1,
            avg_cnn_digit2,
            avg_filtered_digit2,
        ]

        plt.figure(figsize=(9, 5))
        plt.bar(labels, values)
        plt.ylabel("Average Probability of True Digit")
        plt.ylim(0, 1)
        plt.title("Filter Probability Shift on Fixed Shortcut Cases")
        plt.tight_layout()

        save_path = os.path.join(self.figures_dir, "filter_probability_shift.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved {save_path}")

    def print_filter_mechanism_summary(self):
        fixed_rows = [
            row
            for row in self.shortcut_rows
            if str(row["filter_fixed_shortcut"]).lower() == "true"
        ]

        if len(fixed_rows) == 0:
            print("No fixed shortcut cases found.")
            return

        digit1_gains = []
        digit2_gains = []

        for row in fixed_rows:
            digit1_gain = float(row["filtered_digit1_true_prob"]) - float(
                row["cnn_digit1_true_prob"]
            )

            digit2_gain = float(row["filtered_digit2_true_prob"]) - float(
                row["cnn_digit2_true_prob"]
            )

            digit1_gains.append(digit1_gain)
            digit2_gains.append(digit2_gain)

        avg_digit1_gain = sum(digit1_gains) / len(digit1_gains)
        avg_digit2_gain = sum(digit2_gains) / len(digit2_gains)

        print("\nFILTER MECHANISM SUMMARY")
        print(f"Fixed shortcut cases: {len(fixed_rows)}")
        print(f"Average increase in true digit1 probability: {avg_digit1_gain:.4f}")
        print(f"Average increase in true digit2 probability: {avg_digit2_gain:.4f}")

    def plot_confidence_distribution_fixed_vs_unfixed(self):
        fixed_conf = []
        unfixed_conf = []

        for row in self.shortcut_rows:
            true_d1 = int(row["true_digit1"])
            true_d2 = int(row["true_digit2"])

            cnn_d1 = int(row["cnn_pred_digit1"])
            cnn_d2 = int(row["cnn_pred_digit2"])

            wrong_confidences = []

            if true_d1 != cnn_d1:
                wrong_confidences.append(float(row["cnn_digit1_pred_prob"]))

            if true_d2 != cnn_d2:
                wrong_confidences.append(float(row["cnn_digit2_pred_prob"]))

            if len(wrong_confidences) == 0:
                continue

            avg_wrong_confidence = sum(wrong_confidences) / len(wrong_confidences)

            if str(row["filter_fixed_shortcut"]).lower() == "true":
                fixed_conf.append(avg_wrong_confidence)
            else:
                unfixed_conf.append(avg_wrong_confidence)

        if len(fixed_conf) == 0 or len(unfixed_conf) == 0:
            print("Not enough fixed/unfixed cases for confidence distribution plot.")
            return

        fixed_mean = sum(fixed_conf) / len(fixed_conf)
        unfixed_mean = sum(unfixed_conf) / len(unfixed_conf)

        plt.figure(figsize=(8, 5))

        plt.hist(
            fixed_conf,
            bins=15,
            alpha=0.6,
            label=f"Fixed by filter (n={len(fixed_conf)})"
        )

        plt.hist(
            unfixed_conf,
            bins=15,
            alpha=0.6,
            label=f"Not fixed by filter (n={len(unfixed_conf)})"
        )

        plt.axvline(
            fixed_mean,
            linestyle="--",
            linewidth=2,
            label=f"Fixed mean = {fixed_mean:.3f}"
        )

        plt.axvline(
            unfixed_mean,
            linestyle="--",
            linewidth=2,
            label=f"Not fixed mean = {unfixed_mean:.3f}"
        )

        plt.xlabel("CNN Confidence on Wrong Predicted Digit")
        plt.ylabel("Number of Shortcut Cases")
        plt.title("CNN Confidence in Fixed vs Unfixed Shortcut Cases")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(
            self.figures_dir,
            "confidence_distribution_fixed_vs_unfixed.png"
        )

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved {save_path}")
        print(f"Fixed mean confidence: {fixed_mean:.4f}")
        print(f"Not fixed mean confidence: {unfixed_mean:.4f}")

    def generate_all(self):
        self.plot_accuracy_comparison()
        self.plot_shortcut_count_comparison()
        self.plot_shortcut_frequency_comparison()
        self.plot_top_shortcut_confusions()
        self.plot_filter_probability_shift()
        self.plot_confidence_distribution_fixed_vs_unfixed()
        self.print_filter_mechanism_summary()


if __name__ == "__main__":
    visualizer = ResultVisualizer()
    visualizer.generate_all()
