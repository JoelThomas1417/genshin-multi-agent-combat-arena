"""Simplified elemental auras and reactions for the combat arena.

These are transparent simulator assumptions, not exact Genshin Impact rules.
An elemental hit applies an aura for three turns. A compatible second element
consumes that aura and triggers one reaction. Physical damage never creates or
consumes an aura. Unsupported element pairs replace the previous aura.
"""

from copy import deepcopy


AURA_ELEMENTS = {"Pyro", "Hydro", "Electro", "Cryo", "Dendro", "Anemo", "Geo"}

# Pair order does not matter. Damage multipliers apply after normal defense and
# weather calculations. Frozen and Superconduct also add a tactical effect.
REACTIONS = {
    frozenset(("Pyro", "Hydro")): {
        "name": "Vaporize", "damage_multiplier": 1.50,
        "description": "The triggering hit deals 50% more damage.",
    },
    frozenset(("Pyro", "Cryo")): {
        "name": "Melt", "damage_multiplier": 1.50,
        "description": "The triggering hit deals 50% more damage.",
    },
    frozenset(("Pyro", "Electro")): {
        "name": "Overloaded", "damage_multiplier": 1.35,
        "description": "The triggering hit deals 35% more damage.",
    },
    frozenset(("Hydro", "Electro")): {
        "name": "Electro-Charged", "damage_multiplier": 1.25,
        "description": "The triggering hit deals 25% more damage.",
    },
    frozenset(("Hydro", "Cryo")): {
        "name": "Frozen", "damage_multiplier": 1.00, "skip_actions": 1,
        "description": "The target loses its next action.",
    },
    frozenset(("Electro", "Cryo")): {
        "name": "Superconduct", "damage_multiplier": 1.15,
        "defense_reduction": 0.20, "debuff_duration": 3,
        "description": "The hit deals 15% more damage and target DEF falls 20% for three turns.",
    },
}


class ElementalSystem:
    """Track public auras and reaction effects without owning fighter HP."""

    def __init__(self, enabled=False, aura_duration=3):
        if type(enabled) is not bool:
            raise ValueError("reactions_enabled must be True or False.")
        if type(aura_duration) is not int or aura_duration < 1:
            raise ValueError("aura_duration must be a positive whole number.")
        self.enabled = enabled
        self.aura_duration = aura_duration
        self.auras = {}
        self.skipped_actions = {}
        self.defense_debuffs = {}

    def _active_aura(self, target_id, turn):
        aura = self.auras.get(target_id)
        return aura if aura is not None and turn < aura["expires_on"] else None

    def preview(self, target_id, incoming_element, turn):
        """Describe the result of a hit without changing state."""
        result = {
            "reaction": None, "damage_multiplier": 1.0,
            "aura_before": None, "aura_after": None,
        }
        if not self.enabled or incoming_element not in AURA_ELEMENTS:
            return result
        aura = self._active_aura(target_id, turn)
        result["aura_before"] = aura["element"] if aura else None
        if aura is None or aura["element"] == incoming_element:
            result["aura_after"] = incoming_element
            return result
        rule = REACTIONS.get(frozenset((aura["element"], incoming_element)))
        if rule is None:
            result["aura_after"] = incoming_element
            return result
        result.update(deepcopy(rule))
        result["reaction"] = rule["name"]
        result["aura_after"] = None
        return result

    def apply(self, target_id, incoming_element, turn):
        """Apply an elemental hit and return the resulting public description."""
        result = self.preview(target_id, incoming_element, turn)
        if not self.enabled or incoming_element not in AURA_ELEMENTS:
            return result
        if result["reaction"]:
            self.auras.pop(target_id, None)
            if result.get("skip_actions"):
                self.skipped_actions[target_id] = self.skipped_actions.get(target_id, 0) + result["skip_actions"]
            if result.get("defense_reduction"):
                self.defense_debuffs[target_id] = {
                    "fraction": result["defense_reduction"],
                    "expires_on": turn + result["debuff_duration"],
                    "source": result["reaction"],
                }
        else:
            self.auras[target_id] = {"element": incoming_element, "expires_on": turn + self.aura_duration}
        return result

    def defense_multiplier(self, target_id, turn):
        debuff = self.defense_debuffs.get(target_id)
        return 1.0 - debuff["fraction"] if debuff is not None and turn < debuff["expires_on"] else 1.0

    def consume_skipped_action(self, target_id):
        remaining = self.skipped_actions.get(target_id, 0)
        if remaining <= 0:
            return False
        if remaining == 1:
            self.skipped_actions.pop(target_id, None)
        else:
            self.skipped_actions[target_id] = remaining - 1
        return True

    def advance(self, turn):
        """Expire timed auras/debuffs and return public expiry messages."""
        messages = []
        for target_id, aura in list(self.auras.items()):
            if turn >= aura["expires_on"]:
                messages.append({"target": target_id, "effect": "aura", "name": aura["element"],
                                 "message": f"{target_id}'s {aura['element']} aura expires."})
                del self.auras[target_id]
        for target_id, debuff in list(self.defense_debuffs.items()):
            if turn >= debuff["expires_on"]:
                messages.append({"target": target_id, "effect": "defense_debuff", "name": debuff["source"],
                                 "message": f"{target_id}'s {debuff['source']} defense reduction expires."})
                del self.defense_debuffs[target_id]
        return messages

    def clear_target(self, target_id):
        self.auras.pop(target_id, None)
        self.skipped_actions.pop(target_id, None)
        self.defense_debuffs.pop(target_id, None)

    def target_state(self, target_id, turn):
        aura = self._active_aura(target_id, turn)
        debuff = self.defense_debuffs.get(target_id)
        if debuff is not None and turn >= debuff["expires_on"]:
            debuff = None
        return {
            "aura": deepcopy(aura),
            "actions_to_skip": self.skipped_actions.get(target_id, 0),
            "defense_reduction": deepcopy(debuff),
        }

    def configuration(self):
        return {
            "enabled": self.enabled,
            "aura_duration": self.aura_duration,
            "physical_applies_aura": False,
            "reactions": {" + ".join(sorted(pair)): deepcopy(rule)
                          for pair, rule in REACTIONS.items()},
        }
