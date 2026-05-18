# Reward function for metaphorical emotion GRPO training on YesBut.
#
# Format reward: response must contain <caption>...</caption>, <metaphor>...</metaphor>,
# <think>...</think>, <answer>...</answer> tags in order, each appearing exactly once.
# Accuracy reward: predicted emotion (in <answer>) maps to the same sentiment polarity
# (positive / negative / neutral) as the ground truth emotion label.

import re
from typing import Any


REWARD_NAME = "emotion"
REWARD_TYPE = "sequential"


POSITIVE_EMOTIONS = {"happiness", "love", "surprise", "positive"}
NEGATIVE_EMOTIONS = {"anger", "sorrow", "fear", "hate", "negative"}
NEUTRAL_EMOTIONS = {"neutral"}

_FORMAT_PATTERN = re.compile(
    r"^\s*<caption>.*?</caption>\s*<metaphor>.*?</metaphor>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)
_TAGS = ("caption", "metaphor", "think", "answer")


def _emotion_to_sentiment(text: str) -> str:
    tokens = set(re.sub(r"[^a-z]+", " ", text.lower()).split())
    if tokens & POSITIVE_EMOTIONS:
        return "positive"
    if tokens & NEGATIVE_EMOTIONS:
        return "negative"
    if tokens & NEUTRAL_EMOTIONS:
        return "neutral"
    return "unknown"


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
    pred = _emotion_to_sentiment(match.group(1))
    gold = _emotion_to_sentiment(ground_truth)
    if pred == "unknown" or gold == "unknown":
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
