import torch
import torch.nn.functional as F

from reasoning import predict_digits


class Evaluator:
    def __init__(self, model, dataloader, device, filter_model=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.filter_model = filter_model

    def _get_logits(self, images):
        digit1_logits, digit2_logits = self.model(images)

        if self.filter_model is None:
            return digit1_logits, digit2_logits

        digit1_probs = F.softmax(digit1_logits, dim=1)
        digit2_probs = F.softmax(digit2_logits, dim=1)

        concept_probs = torch.cat([digit1_probs, digit2_probs], dim=1)
        original_logits = torch.cat([digit1_logits, digit2_logits], dim=1)

        return self.filter_model(concept_probs, original_logits)

    def _collect_predictions(self):
        all_digit1_pred = []
        all_digit2_pred = []
        all_digit1_true = []
        all_digit2_true = []
        all_target = []

        self.model.eval()

        if self.filter_model is not None:
            self.filter_model.eval()

        with torch.no_grad():
            for images, concepts, target in self.dataloader:
                images = images.to(self.device)
                concepts = concepts.to(self.device)

                digit1_logits, digit2_logits = self._get_logits(images)
                digit1_pred, digit2_pred = predict_digits(digit1_logits, digit2_logits)

                all_digit1_pred.append(digit1_pred.cpu())
                all_digit2_pred.append(digit2_pred.cpu())
                all_digit1_true.append(concepts[:, 0].cpu())
                all_digit2_true.append(concepts[:, 1].cpu())
                all_target.append(target.cpu())

        return {
            "digit1_pred": torch.cat(all_digit1_pred),
            "digit2_pred": torch.cat(all_digit2_pred),
            "digit1_true": torch.cat(all_digit1_true),
            "digit2_true": torch.cat(all_digit2_true),
            "target": torch.cat(all_target),
        }

    def reasoning_accuracy(self, results):
        reasoning_pred = (results["digit1_pred"] + results["digit2_pred"]) % 2

        correct = (reasoning_pred == results["target"]).sum().item()
        total = results["target"].size(0)

        return correct / total, correct, total

    def concept_accuracy(self, results):
        digit1_correct = results["digit1_pred"] == results["digit1_true"]
        digit2_correct = results["digit2_pred"] == results["digit2_true"]

        correct = digit1_correct.sum().item() + digit2_correct.sum().item()
        total = results["digit1_true"].size(0) * 2

        return correct / total, correct, total

    def shortcut_metrics(self, results):
        reasoning_pred = (results["digit1_pred"] + results["digit2_pred"]) % 2

        reasoning_correct = reasoning_pred == results["target"]

        concepts_correct = (results["digit1_pred"] == results["digit1_true"]) & (
            results["digit2_pred"] == results["digit2_true"]
        )

        shortcut_cases = reasoning_correct & (~concepts_correct)

        shortcut_count = shortcut_cases.sum().item()
        correct_reasoning_count = reasoning_correct.sum().item()
        total = results["target"].size(0)

        shortcut_frequency = shortcut_count / total
        shortcut_rate_given_correct = shortcut_count / correct_reasoning_count

        return {
            "shortcut_count": shortcut_count,
            "shortcut_frequency": shortcut_frequency,
            "shortcut_rate_given_correct": shortcut_rate_given_correct,
        }

    def filter_change_analysis(self):
        if self.filter_model is None:
            return None

        fixed = 0
        broken = 0
        unchanged_correct = 0
        unchanged_wrong = 0
        total = 0

        self.model.eval()
        self.filter_model.eval()

        with torch.no_grad():
            for images, concepts, _ in self.dataloader:
                images = images.to(self.device)
                concepts = concepts.to(self.device)

                digit1_true = concepts[:, 0]
                digit2_true = concepts[:, 1]

                cnn_digit1_logits, cnn_digit2_logits = self.model(images)
                cnn_digit1_pred, cnn_digit2_pred = predict_digits(
                    cnn_digit1_logits, cnn_digit2_logits
                )

                filtered_digit1_logits, filtered_digit2_logits = self._get_logits(
                    images
                )
                filtered_digit1_pred, filtered_digit2_pred = predict_digits(
                    filtered_digit1_logits, filtered_digit2_logits
                )

                cnn_correct = (cnn_digit1_pred == digit1_true) & (
                    cnn_digit2_pred == digit2_true
                )

                filter_correct = (filtered_digit1_pred == digit1_true) & (
                    filtered_digit2_pred == digit2_true
                )

                fixed += ((~cnn_correct) & filter_correct).sum().item()
                broken += (cnn_correct & (~filter_correct)).sum().item()
                unchanged_correct += (cnn_correct & filter_correct).sum().item()
                unchanged_wrong += ((~cnn_correct) & (~filter_correct)).sum().item()

                total += concepts.size(0)

        return {
            "fixed_cases": fixed,
            "broken_cases": broken,
            "unchanged_correct": unchanged_correct,
            "unchanged_wrong": unchanged_wrong,
            "total": total,
            "net_gain": fixed - broken,
        }

    def evaluate_all(self):
        results = self._collect_predictions()

        reasoning_acc, reasoning_correct, reasoning_total = self.reasoning_accuracy(
            results
        )
        concept_acc, concept_correct, concept_total = self.concept_accuracy(results)
        shortcut = self.shortcut_metrics(results)

        return {
            "reasoning_accuracy": reasoning_acc,
            "reasoning_correct": reasoning_correct,
            "reasoning_total": reasoning_total,
            "concept_accuracy": concept_acc,
            "concept_correct": concept_correct,
            "concept_total": concept_total,
            **shortcut,
        }
