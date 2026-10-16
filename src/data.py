import re
from typing import Dict, List

from .config import Dataset


def parse_equals_input(text: str) -> List[Dict[str, str]]:
    """Parse a multi-example string separated by '===' into role dictionaries."""
    if not text or not text.strip():
        return []

    raw_blocks = re.split(r"\n?===+\n?", text.strip())
    examples: List[Dict[str, str]] = []
    current_system = None

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        segments = re.split(r"(?=^\s*=\w+=)", block, flags=re.MULTILINE)
        system, user, assistant = None, None, None
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            match = re.match(r"^\s*=(\w+)=\s*(.*)", segment, re.DOTALL)
            if not match:
                continue
            role = match.group(1).upper()
            content = match.group(2).strip()
            if role == "SYSTEM":
                system = content
                current_system = content
            elif role == "USER":
                user = content
            elif role == "ASSISTANT":
                assistant = content

        if not system and current_system:
            system = current_system
        if user and assistant:
            examples.append({"system": system or "", "user": user, "assistant": assistant})

    return examples


def build_dataset_from_roles(parsed: List[Dict[str, str]]):
    """Convert parsed role dictionaries into a Hugging Face dataset."""
    if Dataset is None:
        raise ImportError("The 'datasets' library is required. Install with: pip install datasets")

    conversations = []
    for ex in parsed:
        messages = []
        if ex.get("system"):
            messages.append({"role": "system", "content": ex["system"]})
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
        conversations.append({"messages": messages})
    return Dataset.from_list(conversations)


def parse_training_file(path: str) -> List[Dict[str, str]]:
    """Parse a simple input=output data file into structured training rows."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line:
                continue
            input_value, output_value = line.split("=", 1)
            input_value = input_value.strip()
            output_value = output_value.strip()
            if input_value and output_value:
                rows.append({"input": input_value, "output": output_value})
    return rows


__all__ = ["parse_equals_input", "build_dataset_from_roles", "parse_training_file"]
