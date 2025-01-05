from transformers import AutoModelForCausalLM, LogitsProcessor
import torch
import torch.nn.functional as F

class EntropyTrackingGenerator(LogitsProcessor):
    def __init__(self):
        self.entropy_history = []

    def __call__(self, input_ids, scores):
        probs = F.softmax(scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        self.entropy_history.append(entropy.mean().item())
        return scores

    def average_entropy(self):
        if not self.entropy_history:
            return 0.0
        return sum(self.entropy_history) / len(self.entropy_history)