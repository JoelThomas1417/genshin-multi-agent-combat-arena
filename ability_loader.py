"""Attach simplified abilities to characters from your prepared CSV.

Keep this file, abilities.json, and data_loader.py in the same folder.
The loader uses characters_prepared.csv if present, otherwise genshin_impact.csv.
Only characters configured in abilities.json enter the starter roster.

Run:
    python ability_loader.py
    python ability_loader.py --character noelle
    python ability_loader.py --csv "path/to/characters_prepared.csv"

Use in the future battle engine:
    from ability_loader import load_combat_roster
    rules, roster = load_combat_roster()
    noelle = roster["noelle"]
    shield_amount = noelle["lvl_90_DEF"] * noelle["abilities"]["skill"]["multiplier"]

This module loads and validates definitions; it does not run a battle.
Uses only Python's standard library and your existing data_loader.py.
"""

import argparse
import json
import math
from pathlib import Path

from data_loader import load_characters


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ABILITIES = PROJECT_DIR / "abilities.json"
ABILITY_SLOTS = ("normal_attack", "skill", "burst")
SCALING_STATS = {"lvl_90_HP", "lvl_90_ATK", "lvl_90_DEF"}
DAMAGE_ELEMENTS = {"Physical", "Anemo", "Cryo", "Dendro", "Electro", "Geo", "Hydro", "Pyro"}


def require_integer(value, label, minimum=0):
    """Reject missing, fractional, boolean, or negative integer settings."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be a whole number of at least {minimum}.")


def require_positive_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite number greater than zero.")


def validate_ability(ability, label, maximum_energy):
    """Check that an ability has enough information for the battle engine."""
    if not isinstance(ability.get("name"), str) or not ability["name"].strip():
        raise ValueError(f"{label} needs a name.")
    for key in ("range_cells", "cooldown_turns", "energy_cost", "energy_gain", "duration_turns"):
        require_integer(ability.get(key), f"{label}.{key}")
    if ability["energy_cost"] > maximum_energy:
        raise ValueError(f"{label}.energy_cost exceeds maximum_energy.")

    target = ability.get("target")
    if target not in ("enemy", "ally", "self"):
        raise ValueError(f"{label}.target must be enemy, ally, or self.")
    if target == "self" and ability["range_cells"] != 0:
        raise ValueError(f"{label}: self-targeted abilities must use range_cells 0.")

    effect = ability.get("effect_type")
    if effect not in ("damage", "heal", "shield", "buff"):
        raise ValueError(f"{label} has an unsupported effect_type.")
    if effect == "damage":
        if target != "enemy":
            raise ValueError(f"{label}: starter damage abilities must target an enemy.")
        if ability.get("element") not in DAMAGE_ELEMENTS:
            raise ValueError(f"{label} needs a supported damage element.")
    elif target not in ("ally", "self"):
        raise ValueError(f"{label}: healing, shields, and buffs must target an ally or self.")

    if effect == "buff":
        if ability.get("buff_stat") not in SCALING_STATS:
            raise ValueError(f"{label}.buff_stat must be a supported HP, ATK, or DEF field.")
        # HP buffs need extra maximum-HP rules, so the starter set uses ATK/DEF.
        if ability["buff_stat"] == "lvl_90_HP":
            raise ValueError(f"{label}: starter buffs support ATK and DEF only.")
        require_positive_number(ability.get("bonus_fraction"), f"{label}.bonus_fraction")
    else:
        if ability.get("scaling_stat") not in SCALING_STATS:
            raise ValueError(f"{label}.scaling_stat must be a supported HP, ATK, or DEF field.")
        require_positive_number(ability.get("multiplier"), f"{label}.multiplier")

    if effect in ("shield", "buff") and ability["duration_turns"] < 1:
        raise ValueError(f"{label}: shields and buffs need a positive duration_turns.")
    if effect in ("damage", "heal") and ability["duration_turns"] != 0:
        raise ValueError(f"{label}: starter damage and healing are immediate (duration_turns 0).")


def load_combat_roster(csv_path=None, abilities_path=None):
    """Return (rules, roster_by_name), with CSV statistics and resolved abilities.

    The returned rules include energy settings, units, and the ruleset label.
    Each roster record retains the fields provided by data_loader and adds
    movement_range plus three abilities. This function writes no files.
    """
    if csv_path is None:
        prepared = PROJECT_DIR / "characters_prepared.csv"
        csv_path = prepared if prepared.exists() else PROJECT_DIR / "genshin_impact.csv"
    config_path = Path(abilities_path) if abilities_path is not None else DEFAULT_ABILITIES

    with config_path.open("r", encoding="utf-8-sig") as source:
        config = json.load(source)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("abilities.json must be an object using schema_version 1.")
    for key in ("rules", "units", "ability_defaults", "characters"):
        if not isinstance(config.get(key), dict) or not config[key]:
            raise ValueError(f"abilities.json needs a nonempty {key} object.")

    rules = dict(config["rules"])
    require_integer(rules.get("maximum_energy"), "maximum_energy", minimum=1)
    require_integer(rules.get("starting_energy"), "starting_energy")
    require_integer(rules.get("starting_shield"), "starting_shield")
    if rules["starting_energy"] > rules["maximum_energy"]:
        raise ValueError("starting_energy cannot exceed maximum_energy.")
    for slot in ABILITY_SLOTS:
        if not isinstance(config["ability_defaults"].get(slot), dict):
            raise ValueError(f"ability_defaults needs an object for {slot}.")

    characters = {character["name"]: character for character in load_characters(csv_path)}
    roster = {}
    for name, definition in config["characters"].items():
        if name not in characters:
            raise ValueError(f"Configured character {name!r} was not found in the CSV.")
        if not isinstance(definition, dict):
            raise ValueError(f"{name} must have an ability-definition object.")
        require_integer(definition.get("movement_range"), f"{name}.movement_range", minimum=1)

        abilities = {}
        for slot in ABILITY_SLOTS:
            if not isinstance(definition.get(slot), dict):
                raise ValueError(f"{name} needs a {slot} definition.")
            # Character-specific settings override the shared defaults.
            ability = {**config["ability_defaults"][slot], **definition[slot]}
            validate_ability(ability, f"{name}.{slot}", rules["maximum_energy"])
            abilities[slot] = ability

        roster[name] = {
            **characters[name],
            "movement_range": definition["movement_range"],
            "abilities": abilities,
        }

    rules["ruleset_name"] = config.get("ruleset_name", "Starter arena abilities")
    rules["description"] = config.get("description", "Custom simulator rules.")
    rules["units"] = config["units"]
    return rules, roster


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="Prepared or original character CSV")
    parser.add_argument("--abilities", type=Path, default=DEFAULT_ABILITIES, help="Ability configuration JSON")
    parser.add_argument("--character", help="Print one character's complete setup, for example noelle")
    args = parser.parse_args()
    try:
        rules, roster = load_combat_roster(args.csv, args.abilities)
        if args.character is not None and args.character not in roster:
            raise ValueError(f"Choose one of these configured characters: {', '.join(roster)}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Loaded {len(roster)} characters with normal attacks, skills, and bursts.")
    print("Ability values are custom simulator assumptions.")
    print(f"Starting energy: {rules['starting_energy']} / {rules['maximum_energy']}")
    if args.character is not None:
        print(json.dumps(roster[args.character], indent=2))
    else:
        for name, character in roster.items():
            skill = character["abilities"]["skill"]
            burst = character["abilities"]["burst"]
            print(f"  {name}: skill={skill['effect_type']}, burst={burst['effect_type']}")


if __name__ == "__main__":
    main()
