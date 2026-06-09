from pathlib import Path


def derive_type(label: str) -> str:
    if len(label) == 1:
        return "letter"
    if " " in label:
        return "sentence"
    return "word"


def load_prompts(path: str = "prompts.txt") -> list[tuple[str, str]]:
    """Read prompts.txt -> list of (label, label_type). Type is auto-derived."""
    lines = Path(path).read_text().splitlines()
    prompts = []
    for line in lines:
        label = line.strip()
        if not label or label.startswith("#"):
            continue
        prompts.append((label, derive_type(label)))
    return prompts
