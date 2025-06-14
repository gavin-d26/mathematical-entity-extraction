"""
Utility script to convert prediction output to pandas DataFrame format (like train/val JSON).
"""

import json
import argparse
import uuid
from typing import Dict, List


def convert_predictions_to_pandas_format(
    predictions_file: str, output_file: str
) -> None:
    """
    Convert predictions JSON to pandas DataFrame format like train/val JSON.

    Args:
        predictions_file: Path to the predictions JSON file
        output_file: Path to save the converted JSON file
    """
    # Load predictions
    with open(predictions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data["predictions"]

    # Convert to pandas DataFrame format
    pandas_format = {
        "annoid": {},
        "fileid": {},
        "start": {},
        "end": {},
        "tag": {},
        "text": {},
    }

    row_idx = 0

    for file_id, spans in predictions.items():
        for span in spans:
            # Generate a unique annotation ID
            annoid = str(uuid.uuid4())

            pandas_format["annoid"][str(row_idx)] = annoid
            pandas_format["fileid"][str(row_idx)] = file_id
            pandas_format["start"][str(row_idx)] = span["start"]
            pandas_format["end"][str(row_idx)] = span["end"]
            pandas_format["tag"][str(row_idx)] = span["tag"]
            pandas_format["text"][str(row_idx)] = span["text"]

            row_idx += 1

    # Save converted format
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pandas_format, f, indent=2, ensure_ascii=False)

    print(f"Converted {row_idx} predictions to pandas format")
    print(f"Saved to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert predictions to pandas DataFrame format"
    )
    parser.add_argument("predictions_file", help="Path to predictions JSON file")
    parser.add_argument(
        "--output_file",
        default="predictions_pandas_format.json",
        help="Output file name (default: predictions_pandas_format.json)",
    )

    args = parser.parse_args()

    convert_predictions_to_pandas_format(args.predictions_file, args.output_file)


if __name__ == "__main__":
    main()
