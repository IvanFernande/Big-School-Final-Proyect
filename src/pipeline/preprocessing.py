import re


def clean_text(text: str, preserve_newlines: bool = False) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
