import math
import re
from collections import Counter

import pandas as pd

from config import PROCESSED_CSV


def clean_text(text):
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None
    text = str(text)
    text = text.lower()
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text if text else None


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    print(df.dtypes)

    df["findings"] = df["findings"].apply(clean_text)
    df["impression"] = df["impression"].apply(clean_text)
    df["indication"] = df["indication"].apply(clean_text)

    print("After cleaning — sample impression:")
    print(df["impression"].iloc[0])
    print("\nAfter cleaning — sample findings:")
    print(df["findings"].iloc[0])

    df["input_text"] = df["impression"].fillna(df["findings"])
    assert df["input_text"].isna().sum() == 0, "Some rows still have no input text!"

    print(f"Input text ready for {len(df)} reports")
    print(f"\nSample input:\n{df['input_text'].iloc[0]}")

    all_text = " ".join(df["input_text"].dropna().tolist())
    words = re.findall(r"\b[a-z]{4,}\b", all_text)
    freq = Counter(words)
    stopwords = {
        "with",
        "and",
        "the",
        "this",
        "that",
        "from",
        "have",
        "been",
        "were",
        "there",
        "their",
        "also",
        "which",
        "will",
    }
    medical_freq = {w: c for w, c in freq.most_common(100) if w not in stopwords}
    for word, count in list(medical_freq.items())[:40]:
        print(f"{count:>5}  {word}")


if __name__ == "__main__":
    main()
