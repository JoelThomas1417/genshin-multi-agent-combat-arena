"""Calculate the seven-part competition score for a completed battle.

Each component is normalized to 0..1 and combined using the challenge weights:
25% survival, 20% objective control, 15% resource management, 15% team
synergy, 10% damage contribution, 10% adaptability, and 5% risk efficiency.

The resulting weighted_score is an evaluation score, separate from the arena's
winner rule. For zero-sum learning, relative_reward(team) is its weighted score
minus the opponent's score and therefore lies in -1..1.

Starter measurement definitions (custom, inspectable simulator assumptions):
* survival: remaining team HP / total maximum HP.
* objective_control: control points / target, or opposing knockout fraction for
  elimination matches.
* resource_management: 70% valid-action rate + 30% ending-energy fraction.
* team_synergy: average active synergy score during the match / 100.
* damage_contribution: HP damage dealt / opposing total maximum HP, capped at 1.
* adaptability: avoidance of forbidden burst/shield actions during arena events;
  0.5 is neutral when no event creates an opportunity to demonstrate it.
* risk_efficiency: damage dealt / (damage dealt + damage received); 0.5 when
  neither team deals HP damage.
"""


WEIGHTS = {
    "survival": 0.25,
    "objective_control": 0.20,
    "resource_management": 0.15,
    "team_synergy": 0.15,
    "damage_contribution": 0.10,
    "adaptability": 0.10,
    "risk_efficiency": 0.05,
}


def _clip(value):
    return max(0.0, min(1.0, float(value)))


def _team(uid):
    return uid[0] if isinstance(uid, str) and len(uid) > 2 and uid[1] == ":" and uid[0] in "AB" else None


def calculate_rewards(battle):
    """Return component scores, weighted scores, and relative rewards."""
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("Reward weights must sum to 1.")
    teams = ("A", "B")
    opponent = {"A": "B", "B": "A"}
    fighters = {team: [f for f in battle.fighters.values() if f.team == team] for team in teams}
    max_hp = {team: sum(f.max_hp for f in fighters[team]) for team in teams}
    remaining_hp = {team: sum(f.hp for f in fighters[team]) for team in teams}
    ending_energy = {team: sum(f.energy for f in fighters[team]) for team in teams}
    damage = {team: 0.0 for team in teams}
    invalid = {team: 0 for team in teams}
    planned = {team: 0 for team in teams}
    event_opportunities = {team: 0 for team in teams}
    event_violations = {team: 0 for team in teams}
    synergy_samples = {team: [] for team in teams}

    for turn in battle.history:
        active_event = turn["before"]["arena"].get("active_event")
        event_name = active_event.get("name") if active_event else None
        for team in teams:
            synergy_samples[team].append(turn["after"]["synergy"][team]["active_score"])
        for uid, action in turn["actions"].items():
            team = _team(uid)
            if team is None or not isinstance(action, dict):
                continue
            planned[team] += 1
            if event_name in ("energy_blackout", "shield_nullification"):
                event_opportunities[team] += 1
                if action.get("kind") == "ability":
                    slot = action.get("slot")
                    ability = battle.fighters[uid].data["abilities"].get(slot, {})
                    if event_name == "energy_blackout" and slot == "burst":
                        event_violations[team] += 1
                    elif event_name == "shield_nullification" and ability.get("effect_type") == "shield":
                        event_violations[team] += 1
        for event in turn["events"]:
            team = _team(event.get("actor"))
            if team:
                damage[team] += float(event.get("hp_damage", 0))
                invalid[team] += int(event.get("kind") == "invalid")

    components = {}
    for team in teams:
        enemy = opponent[team]
        survival = remaining_hp[team] / max_hp[team]
        if battle.objective.mode == "control_zone":
            objective = battle.objective.scores[team] / battle.objective.target_score
        else:
            objective = 1.0 - len(battle.living(enemy)) / len(fighters[enemy])
        valid_rate = 1.0 - invalid[team] / planned[team] if planned[team] else 0.0
        energy_fraction = ending_energy[team] / (battle.rules["maximum_energy"] * len(fighters[team]))
        resource = 0.70 * valid_rate + 0.30 * energy_fraction
        if synergy_samples[team]:
            synergy = sum(synergy_samples[team]) / len(synergy_samples[team]) / 100.0
        else:
            synergy = battle.initial_state["synergy"][team]["active_score"] / 100.0
        damage_component = damage[team] / max_hp[enemy]
        opportunities = event_opportunities[team]
        adaptability = 1.0 - event_violations[team] / opportunities if opportunities else 0.5
        combined_damage = damage[team] + damage[enemy]
        risk = damage[team] / combined_damage if combined_damage else 0.5
        components[team] = {
            "survival": _clip(survival),
            "objective_control": _clip(objective),
            "resource_management": _clip(resource),
            "team_synergy": _clip(synergy),
            "damage_contribution": _clip(damage_component),
            "adaptability": _clip(adaptability),
            "risk_efficiency": _clip(risk),
        }

    weighted = {team: sum(WEIGHTS[name] * value for name, value in components[team].items())
                for team in teams}
    return {
        "weights": dict(WEIGHTS),
        "teams": {
            team: {
                "components": {name: round(value, 6) for name, value in components[team].items()},
                "weighted_score": round(weighted[team], 6),
                "relative_reward": round(weighted[team] - weighted[opponent[team]], 6),
            }
            for team in teams
        },
    }
