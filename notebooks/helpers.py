from pathlib import Path

from IPython.display import Markdown, display


def display_yaml(path_str: str) -> None:
    yaml_text = Path(path_str).read_text(encoding="utf-8")
    display(Markdown(f"```yaml\n{yaml_text}\n```"))
