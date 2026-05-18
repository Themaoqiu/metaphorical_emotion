# Reward function for metaphorical emotion GRPO training on YesBut.
#
# Format reward: response must contain <caption>...</caption>, <metaphor>...</metaphor>,
# <think>...</think>, <answer>...</answer> tags in order, each appearing exactly once.
# Accuracy reward: predicted emotion (in <answer>) must match the ground truth
# emotion label exactly after light normalization.

import re
from typing import Any


REWARD_NAME = "emotion"
REWARD_TYPE = "sequential"


_FORMAT_PATTERN = re.compile(
    r"^\s*<caption>.*?</caption>\s*<metaphor>.*?</metaphor>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)
_TAGS = ("caption", "metaphor", "think", "answer")


def _normalize_emotion(text: str) -> str:
    return " ".join(re.sub(r"[^a-z]+", " ", text.lower()).split())


def format_reward(response: str) -> float:
    if not _FORMAT_PATTERN.fullmatch(response):
        return 0.0
    for tag in _TAGS:
        if len(re.findall(rf"<{tag}>", response)) != 1:
            return 0.0
        if len(re.findall(rf"</{tag}>", response)) != 1:
            return 0.0
    return 1.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return 0.0
    pred = _normalize_emotion(match.group(1))
    gold = _normalize_emotion(ground_truth)
    if not pred or not gold:
        return 0.0
    return 1.0 if pred == gold else 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
    response = reward_input["response"]
    ground_truth = reward_input["ground_truth"]
    fmt = format_reward(response)
    acc = accuracy_reward(response, ground_truth)
    return {
        "overall": format_weight * fmt + (1.0 - format_weight) * acc,
        "format": fmt,
        "accuracy": acc,
    }
