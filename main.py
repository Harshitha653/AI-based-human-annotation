import glob
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from config import ECGEN_XML_DIR, PROCESSED_CSV


def parse_report(xml_path: Path) -> dict:
    with open(xml_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    report_id = soup.find("uId")["id"]
    pmc_id = soup.find("pmcId")["id"]

    def get_section(label: str):
        tag = soup.find("AbstractText", {"Label": label})
        if tag and tag.text and tag.text.strip():
            return tag.text.strip()
        return None

    return {
        "report_id": report_id,
        "pmc_id": pmc_id,
        "comparison": get_section("COMPARISON"),
        "indication": get_section("INDICATION"),
        "findings": get_section("FINDINGS"),
        "impression": get_section("IMPRESSION"),
    }


def clean_text(text):
    if text is None:
        return None
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def main() -> None:
    xml_files = sorted(glob.glob(str(ECGEN_XML_DIR / "*.xml")))
    if not xml_files:
        print(f"No XML files found under {ECGEN_XML_DIR}")
        return

    records = [parse_report(Path(p)) for p in xml_files]
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} reports")
    print(df[["report_id", "findings", "impression"]].head(3))

    df["findings"] = df["findings"].apply(clean_text)
    df["impression"] = df["impression"].apply(clean_text)

    df = df.dropna(subset=["findings", "impression"], how="all")

    df["impression"] = df["impression"].apply(
        lambda x: re.sub(r"^\d+\.\s*", "", x) if x else x
    )

    df["input_text"] = df["impression"].fillna(df["findings"])

    print(f"Usable reports: {len(df)}")
    print(f"Missing impression: {df['impression'].isna().sum()}")
    print(f"Missing findings:   {df['findings'].isna().sum()}")

    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Saved → {PROCESSED_CSV}")


if __name__ == "__main__":
    main()
