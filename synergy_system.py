"""Load and apply transparent team-synergy rules.

The rules are custom simulator assumptions stored in synergies.json. Pair scores
measure composition quality; effect bonuses change combat only when synergy is
enabled. Environmental events can suppress a pair or individual effects.
"""

from copy import deepcopy
import json
import math
from pathlib import Path


DEFAULT_SYNERGIES = Path(__file__).resolve().with_name("synergies.json")
NUMERIC_EFFECTS = {"atk_bonus", "def_bonus", "shield_bonus", "healing_bonus",
                   "energy_gain_bonus", "reaction_damage_bonus"}
SUPPORTED_ELEMENTS = {"Physical", "Anemo", "Cryo", "Dendro", "Electro", "Geo", "Hydro", "Pyro"}
SUPPORTED_EVENTS = {"energy_blackout", "shield_nullification"}


def _bonus(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be a finite number from 0 to 1.")


def load_synergy_rules(path=DEFAULT_SYNERGIES):
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != 1 or not isinstance(data.get("pairs"), list):
        raise ValueError("Synergy file must use schema_version 1 and contain a pairs list.")
    seen = set()
    for index, pair in enumerate(data["pairs"], 1):
        label = f"pairs[{index}]"
        names = pair.get("characters")
        if not isinstance(names, list) or len(names) != 2 or any(not isinstance(n, str) or not n for n in names):
            raise ValueError(f"{label}.characters must contain two character names.")
        key = frozenset(names)
        if len(key) != 2 or key in seen:
            raise ValueError(f"{label} must define a unique pair of different characters.")
        seen.add(key)
        if type(pair.get("score")) is not int or pair["score"] < 0:
            raise ValueError(f"{label}.score must be a non-negative whole number.")
        if not isinstance(pair.get("description"), str) or not pair["description"].strip():
            raise ValueError(f"{label}.description is required.")
        effects = pair.get("effects")
        if not isinstance(effects, dict) or not effects:
            raise ValueError(f"{label}.effects must be a non-empty object.")
        unknown = set(effects) - NUMERIC_EFFECTS - {"element_damage_bonus"}
        if unknown:
            raise ValueError(f"{label} has unsupported effects: {sorted(unknown)}")
        for name, value in effects.items():
            if name == "element_damage_bonus":
                if not isinstance(value, dict) or not value or set(value) - SUPPORTED_ELEMENTS:
                    raise ValueError(f"{label}.element_damage_bonus has invalid elements.")
                for element, amount in value.items():
                    _bonus(amount, f"{label}.element_damage_bonus.{element}")
            else:
                _bonus(value, f"{label}.{name}")
        inactive = pair.get("inactive_events", [])
        if not isinstance(inactive, list) or set(inactive) - SUPPORTED_EVENTS:
            raise ValueError(f"{label}.inactive_events contains unsupported events.")
        suppressed = pair.get("suppressed_effects", {})
        if not isinstance(suppressed, dict) or set(suppressed) - SUPPORTED_EVENTS:
            raise ValueError(f"{label}.suppressed_effects contains unsupported events.")
        for event, effect_names in suppressed.items():
            if not isinstance(effect_names, list) or set(effect_names) - set(effects):
                raise ValueError(f"{label}.suppressed_effects.{event} references an unknown effect.")
        if type(pair.get("requires_reactions", False)) is not bool:
            raise ValueError(f"{label}.requires_reactions must be true or false.")
    coverage = data.get("role_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("role_coverage is required.")
    flags = coverage.get("required_flags")
    if not isinstance(flags, list) or not flags or any(not isinstance(flag, str) for flag in flags):
        raise ValueError("role_coverage.required_flags must be a non-empty list.")
    if type(coverage.get("score")) is not int or coverage["score"] < 0:
        raise ValueError("role_coverage.score must be a non-negative whole number.")
    if type(coverage.get("minimum_team_size")) is not int or coverage["minimum_team_size"] < 2:
        raise ValueError("role_coverage.minimum_team_size must be at least two.")
    return data


class SynergySystem:
    def __init__(self, roster, teams, enabled=False, rules_path=DEFAULT_SYNERGIES):
        if type(enabled) is not bool:
            raise ValueError("synergy_enabled must be True or False.")
        self.enabled = enabled
        self.rules_path = Path(rules_path)
        self.rules = load_synergy_rules(self.rules_path)
        self.roster = roster
        self.teams = {team: tuple(names) for team, names in teams.items()}
        for team, names in self.teams.items():
            if team not in ("A", "B") or any(name not in roster for name in names):
                raise ValueError("Synergy teams must use known roster characters on teams A and B.")

    def _event_names(self, arena):
        return {name for name in SUPPORTED_EVENTS if arena.active(name)}

    def _role_coverage(self, team):
        rule = self.rules["role_coverage"]
        names = self.teams[team]
        covered = {flag for flag in rule["required_flags"]
                   if any(self.roster[name].get(flag) == 1 for name in names)}
        active = len(names) >= rule["minimum_team_size"] and covered == set(rule["required_flags"])
        return active, sorted(covered)

    def state(self, team, arena, reactions_enabled):
        names = set(self.teams[team])
        events = self._event_names(arena)
        active_pairs, suppressed_pairs = [], []
        effects = {name: 0.0 for name in NUMERIC_EFFECTS}
        effects["element_damage_bonus"] = {}
        base_score = active_score = 0
        for pair in self.rules["pairs"]:
            if not set(pair["characters"]).issubset(names):
                continue
            base_score += pair["score"]
            reasons = []
            if pair.get("requires_reactions") and not reactions_enabled:
                reasons.append("elemental reactions are disabled")
            blocked = events.intersection(pair.get("inactive_events", []))
            reasons.extend(event.replace("_", " ") for event in sorted(blocked))
            record = {"characters": list(pair["characters"]), "score": pair["score"],
                      "description": pair["description"]}
            if reasons:
                record["reasons"] = reasons
                suppressed_pairs.append(record)
                continue
            active_score += pair["score"]
            active_pairs.append(record)
            suppressed_names = set()
            for event in events:
                suppressed_names.update(pair.get("suppressed_effects", {}).get(event, []))
            for effect, value in pair["effects"].items():
                if effect in suppressed_names:
                    continue
                if effect == "element_damage_bonus":
                    for element, amount in value.items():
                        effects[effect][element] = effects[effect].get(element, 0.0) + amount
                else:
                    effects[effect] += value
        coverage_active, covered_roles = self._role_coverage(team)
        if coverage_active:
            score = self.rules["role_coverage"]["score"]
            base_score += score
            active_score += score
        if not self.enabled:
            active_score = 0
            effects = {name: 0.0 for name in NUMERIC_EFFECTS} | {"element_damage_bonus": {}}
        return {
            "enabled": self.enabled, "characters": list(self.teams[team]),
            "base_score": base_score, "active_score": active_score,
            "active_pairs": active_pairs if self.enabled else [],
            "suppressed_pairs": suppressed_pairs if self.enabled else [],
            "role_coverage": {"active": coverage_active and self.enabled,
                              "covered_flags": covered_roles,
                              "score": self.rules["role_coverage"]["score"] if coverage_active else 0},
            "effects": effects,
        }

    def effects(self, team, arena, reactions_enabled):
        return self.state(team, arena, reactions_enabled)["effects"]

    def stat_multiplier(self, team, stat, arena, reactions_enabled):
        key = {"lvl_90_ATK": "atk_bonus", "lvl_90_DEF": "def_bonus"}.get(stat)
        return 1.0 + (self.effects(team, arena, reactions_enabled)[key] if key else 0.0)

    def element_damage_multiplier(self, team, element, arena, reactions_enabled):
        values = self.effects(team, arena, reactions_enabled)["element_damage_bonus"]
        return 1.0 + values.get(element, 0.0)

    def effect_multiplier(self, team, effect, arena, reactions_enabled):
        key = {"shield": "shield_bonus", "heal": "healing_bonus",
               "energy": "energy_gain_bonus", "reaction": "reaction_damage_bonus"}[effect]
        return 1.0 + self.effects(team, arena, reactions_enabled)[key]

    def configuration(self):
        return {"enabled": self.enabled, "rules_file": self.rules_path.name,
                "schema_version": self.rules["schema_version"], "rules": deepcopy(self.rules)}
