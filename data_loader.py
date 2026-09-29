"""Prepare Genshin character data for the combat simulator.

Uses only Python's standard library; no pip installation is needed.
Place genshin_impact.csv beside this file, then run:

    python data_loader.py

To save the prepared data as a separate CSV:

    python data_loader.py --output characters_prepared.csv

You can also supply a different source path:

    python data_loader.py "path/to/genshin_impact.csv"

From another Python file:

    from data_loader import load_characters
    characters = load_characters()
    roster = {character["name"]: character for character in characters}
    print(roster["noelle"]["role_survivability"])  # 1

Each character is a dictionary with the seven selected source fields and five
role flags. Statistics are positive integers. Role flags are 1 (yes) or 0 (no).
The roles field preserves all role labels as comma-separated text. Popularity,
banner metadata, and the unnamed index are excluded from the returned records.
"""

import argparse
import csv
from pathlib import Path


COMBAT_COLUMNS = (
    "name",
    "element",
    "weapon",
    "lvl_90_HP",
    "lvl_90_ATK",
    "lvl_90_DEF",
    "roles",
)

STAT_COLUMNS = ("lvl_90_HP", "lvl_90_ATK", "lvl_90_DEF")

ROLE_FLAGS = {
    "On-Field": "role_on_field",
    "Off-Field": "role_off_field",
    "DPS": "role_dps",
    "Support": "role_support",
    "Survivability": "role_survivability",
}

OUTPUT_COLUMNS = (*COMBAT_COLUMNS, *ROLE_FLAGS.values())
DEFAULT_CSV = Path(__file__).resolve().with_name("genshin_impact.csv")


def load_characters(csv_path=None):
    """Return a list of validated character dictionaries.

    csv_path defaults to genshin_impact.csv beside this script.
    Missing fields, invalid statistics, duplicate names, and unknown roles
    raise ValueError. File-access errors are reported by Python normally.
    This function reads the source file without changing it.
    """
    source_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    characters = []
    seen_names = set()

    # utf-8-sig also accepts CSV files that have a UTF-8 byte-order mark.
    with source_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        missing_columns = [column for column in COMBAT_COLUMNS if column not in headers]
        if missing_columns:
            raise ValueError("Missing required columns: " + ", ".join(missing_columns))
        if any(headers.count(column) > 1 for column in COMBAT_COLUMNS):
            raise ValueError("A required column appears more than once in the CSV header.")

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"CSV row {row_number} has more values than its header.")

            # Selecting these fields excludes the index and all other metadata.
            character = {}
            for column in COMBAT_COLUMNS:
                value = row.get(column)
                if value is None or not value.strip():
                    raise ValueError(f"CSV row {row_number}: {column} is missing.")
                character[column] = value.strip()

            name_key = character["name"].casefold()
            if name_key in seen_names:
                raise ValueError(f"CSV row {row_number}: duplicate name {character['name']!r}.")
            seen_names.add(name_key)

            # Keep real combat statistics; do not normalize them into game values.
            for column in STAT_COLUMNS:
                try:
                    value = int(character[column])
                except ValueError as error:
                    raise ValueError(
                        f"CSV row {row_number}: {column} must be a positive whole number."
                    ) from error
                if value <= 0:
                    raise ValueError(f"CSV row {row_number}: {column} must be greater than zero.")
                character[column] = value

            # A character may have several roles, so give each role its own flag.
            roles = [role.strip() for role in character["roles"].split(",")]
            unknown_roles = set(roles) - set(ROLE_FLAGS)
            if unknown_roles:
                raise ValueError(
                    f"CSV row {row_number}: unrecognized roles {sorted(unknown_roles)!r}. "
                    "Add their definitions to ROLE_FLAGS before loading this data."
                )
            character["roles"] = ",".join(roles)
            for role, flag_column in ROLE_FLAGS.items():
                character[flag_column] = int(role in roles)

            characters.append(character)

    if not characters:
        raise ValueError("The CSV contains no character records.")
    return characters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV, help="Source CSV path")
    parser.add_argument("--output", type=Path, help="Optional path for a separate prepared CSV")
    args = parser.parse_args()

    try:
        characters = load_characters(args.csv_path)
        if args.output is not None:
            same_path = args.output.resolve() == args.csv_path.resolve()
            same_file = args.output.exists() and args.output.samefile(args.csv_path)
            if same_path or same_file:
                raise ValueError("Choose a different output path to preserve the original dataset.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()
                writer.writerows(characters)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Loaded {len(characters)} characters with {len(OUTPUT_COLUMNS)} prepared fields.")
    print("Role counts (characters can have multiple roles):")
    for role, flag_column in ROLE_FLAGS.items():
        print(f"  {role}: {sum(character[flag_column] for character in characters)}")
    if args.output is not None:
        print(f"Saved prepared data to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
