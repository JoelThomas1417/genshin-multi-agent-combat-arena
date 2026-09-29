"""Weather and stochastic events for the starter combat simulator.

All numerical effects are custom simulator assumptions, not official game rules.
Weather stays fixed for a match. If events are enabled, sample an event every
five turns: 40% no event, 30% energy blackout, 30% shield nullification.

An event beginning on turn T affects T, T+1, and T+2 and ends before turn T+3.
Blackout prevents bursts and all ability energy gain; stored energy is kept.
Shield nullification removes existing shields and prevents new shield abilities.
Ending nullification does not restore the removed shields.

Future events are not scheduled in advance or included in public_state().
The first probabilities are constant independent draws, not a learned forecast.
Visibility, terrain hazards, elemental reactions, and objectives are later work.
"""

from copy import deepcopy
import random


WEATHER_RULES = {
    "clear": {
        "description": "No weather modifiers.",
        "movement_penalty": 0,
        "damage_multipliers": {},
    },
    "blizzard": {
        "description": "Movement is reduced by one cell (minimum one); Cryo damage is increased by 20%.",
        "movement_penalty": 1,
        "damage_multipliers": {"Cryo": 1.20},
    },
    "thunderstorm": {
        "description": "Electro damage is increased by 30%.",
        "movement_penalty": 0,
        "damage_multipliers": {"Electro": 1.30},
    },
}

EVENT_PROBABILITIES = {
    "none": 0.40,
    "energy_blackout": 0.30,
    "shield_nullification": 0.30,
}

EVENT_DESCRIPTIONS = {
    "energy_blackout": "Bursts are unavailable and attacks/skills gain no energy. Stored energy is retained.",
    "shield_nullification": "Existing shields are removed and new shield abilities are unavailable.",
}


class ArenaEnvironment:
    """Keep environmental randomness separate from fighter initiative randomness."""

    def __init__(self, weather="clear", events_enabled=False, seed=42,
                 event_interval=5, event_duration=3):
        if weather not in (*WEATHER_RULES, "random"):
            raise ValueError(f"Unknown weather {weather!r}. Choose clear, blizzard, thunderstorm, or random.")
        if type(events_enabled) is not bool:
            raise ValueError("events_enabled must be True or False.")
        if type(event_interval) is not int or event_interval < 1:
            raise ValueError("event_interval must be a positive whole number.")
        if type(event_duration) is not int or not 1 <= event_duration <= event_interval:
            raise ValueError("event_duration must be between 1 and event_interval.")

        self.requested_weather = weather
        self.weather = random.Random(f"{seed}:arena-weather").choice(tuple(WEATHER_RULES)) if weather == "random" else weather
        self.events_enabled = events_enabled
        self.event_interval = event_interval
        self.event_duration = event_duration
        self.turn = 0
        self.active_event = None
        # This private stream is never included in an agent observation.
        self._event_rng = random.Random(f"{seed}:arena-events")

    def active(self, event_name):
        return self.active_event is not None and self.active_event["name"] == event_name

    def advance(self, turn):
        """Advance exactly one turn and return public announcements."""
        if type(turn) is not int or turn != self.turn + 1:
            raise ValueError("Advance the arena once per turn, in order.")
        self.turn = turn
        announcements = []
        if self.active_event is not None and turn >= self.active_event["expires_on"]:
            name = self.active_event["name"]
            announcements.append({"event": name, "transition": "end", "message": f"{name.replace('_', ' ').title()} ends."})
            self.active_event = None

        if self.events_enabled and turn % self.event_interval == 0:
            draw = self._event_rng.random()
            cumulative = 0.0
            chosen = "none"
            for event_name, probability in EVENT_PROBABILITIES.items():
                cumulative += probability
                if draw < cumulative:
                    chosen = event_name
                    break
            if chosen == "none":
                announcements.append({"event": "none", "transition": "check", "message": "Arena event check: no new event."})
            else:
                self.active_event = {"name": chosen, "started_on": turn, "expires_on": turn + self.event_duration}
                announcements.append({
                    "event": chosen, "transition": "start",
                    "message": f"{chosen.replace('_', ' ').title()} begins for {self.event_duration} turns. {EVENT_DESCRIPTIONS[chosen]}",
                })
        return announcements

    def movement_allowance(self, base_range):
        return max(1, base_range - WEATHER_RULES[self.weather]["movement_penalty"])

    def damage_multiplier(self, element):
        return WEATHER_RULES[self.weather]["damage_multipliers"].get(element, 1.0)

    def energy_gain(self, base_gain):
        return 0 if self.active("energy_blackout") else base_gain

    def ability_block_reason(self, slot, ability):
        if self.active("energy_blackout") and slot == "burst":
            return "energy blackout prevents bursts"
        if self.active("shield_nullification") and ability["effect_type"] == "shield":
            return "shield nullification prevents new shields"
        return None

    def public_state(self):
        """Expose current effects and probabilities, never a future event outcome."""
        active_event = deepcopy(self.active_event)
        if active_event is not None:
            active_event["remaining_turns"] = active_event["expires_on"] - self.turn
            active_event["description"] = EVENT_DESCRIPTIONS[active_event["name"]]
        next_check = ((self.turn // self.event_interval) + 1) * self.event_interval if self.events_enabled else None
        return {
            "weather": self.weather,
            "weather_effects": deepcopy(WEATHER_RULES[self.weather]),
            "terrain": "open_grid",
            "visibility": "full",
            "energy_flux": "blocked" if self.active("energy_blackout") else "normal",
            "hazard_map": [],
            "events_enabled": self.events_enabled,
            "active_event": active_event,
            "event_forecast": {
                "next_check_turn": next_check,
                "probabilities": dict(EVENT_PROBABILITIES) if self.events_enabled else {},
                "basis": "Independent draws at each event check; the actual next outcome is unknown.",
            },
        }

    def configuration(self):
        """Configuration for post-match replay, without random-generator state."""
        return {
            "requested_weather": self.requested_weather,
            "resolved_weather": self.weather,
            "events_enabled": self.events_enabled,
            "event_interval": self.event_interval,
            "event_duration": self.event_duration,
            "event_probabilities": dict(EVENT_PROBABILITIES),
            "weather_rules": deepcopy(WEATHER_RULES),
        }
