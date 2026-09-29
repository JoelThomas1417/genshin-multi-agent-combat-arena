"""Generate a reproducible dataset of combat-arena matches.

Keep this file with the existing simulator files, then run:

    python simulation_runner.py --matches 100 --output simulation_results.csv
    python simulation_runner.py --matches 1000 --seed 2026 --team-size random
    python simulation_runner.py --matches 20 --logs-dir detailed_logs

The runner uses only Python's standard library. It samples disjoint teams,
weather, objectives, controller styles, and battle seeds. Events, elemental
reactions, and synergy are enabled by default. Each CSV row summarizes one
completed match; --logs-dir optionally saves the full JSON turn history.

The transparent base-stat power index is:
    sum(HP / 10 + ATK + DEF)
It is used only to label the lower-stat team and statistical upsets. It does not
change combat and is not claimed to represent official character strength.
"""

import argparse
import csv
import json
from pathlib import Path
import random
import tempfile

from ability_loader import DEFAULT_ABILITIES, PROJECT_DIR, load_combat_roster
from arena_objectives import OBJECTIVE_TYPES
from battle_engine import Battle
from reward_calculator import calculate_rewards
from synergy_system import DEFAULT_SYNERGIES


WEATHERS = ("clear", "blizzard", "thunderstorm")
CONTROLLERS = ("objective", "combat")

FIELDS = (
    "match_id", "runner_seed", "battle_seed", "team_size",
    "team_a", "team_b", "stat_power_a", "stat_power_b", "weaker_team",
    "weather", "objective", "controller_a", "controller_b", "max_turns",
    "events_enabled", "reactions_enabled", "synergy_enabled",
    "winner", "upset", "reason", "turns",
    "reward_survival_a", "reward_survival_b",
    "reward_objective_control_a", "reward_objective_control_b",
    "reward_resource_management_a", "reward_resource_management_b",
    "reward_team_synergy_a", "reward_team_synergy_b",
    "reward_damage_contribution_a", "reward_damage_contribution_b",
    "reward_adaptability_a", "reward_adaptability_b",
    "reward_risk_efficiency_a", "reward_risk_efficiency_b",
    "weighted_score_a", "weighted_score_b", "relative_reward_a", "relative_reward_b",
    "objective_score_a", "objective_score_b", "synergy_score_a", "synergy_score_b",
    "survivors_a", "survivors_b", "survival_fraction_a", "survival_fraction_b",
    "damage_a", "damage_b", "shield_damage_a", "shield_damage_b",
    "healing_a", "healing_b", "energy_spent_a", "energy_spent_b",
    "energy_gained_a", "energy_gained_b", "ending_energy_a", "ending_energy_b",
    "reactions_a", "reactions_b", "energy_blackouts", "shield_nullifications",
    "synergy_changes_a", "synergy_changes_b", "frozen_actions_a", "frozen_actions_b",
    "moves_a", "moves_b", "defends_a", "defends_b",
    "normal_attacks_a", "normal_attacks_b", "skills_a", "skills_b", "bursts_a", "bursts_b",
    "invalid_actions_a", "invalid_actions_b",
)


def team_of(uid):
    return uid[0] if isinstance(uid, str) and len(uid) > 2 and uid[1] == ":" and uid[0] in "AB" else None


def stat_power(roster, names):
    return round(sum(roster[name]["lvl_90_HP"] / 10 + roster[name]["lvl_90_ATK"]
                     + roster[name]["lvl_90_DEF"] for name in names), 2)


def weaker_label(power_a, power_b):
    if power_a < power_b:
        return "A"
    if power_b < power_a:
        return "B"
    return "Tie"


def summarize(battle, match_id, runner_seed, battle_seed, team_a, team_b):
    result = battle.result()
    rewards = calculate_rewards(battle)
    powers = {"A": stat_power(battle.synergy.roster, team_a),
              "B": stat_power(battle.synergy.roster, team_b)}
    weaker = weaker_label(powers["A"], powers["B"])
    totals = {team: {
        "damage": 0.0, "shield_damage": 0.0, "healing": 0.0,
        "energy_spent": 0, "energy_gained": 0, "reactions": 0,
        "synergy_changes": 0, "frozen_actions": 0, "invalid_actions": 0,
        "move": 0, "defend": 0, "normal_attack": 0, "skill": 0, "burst": 0,
    } for team in ("A", "B")}
    arena_events = {"energy_blackout": 0, "shield_nullification": 0}

    for turn in battle.history:
        for uid, action in turn["actions"].items():
            team = team_of(uid)
            if team is None or not isinstance(action, dict):
                continue
            kind = action.get("kind")
            if kind in ("move", "defend"):
                totals[team][kind] += 1
            elif kind == "ability" and action.get("slot") in ("normal_attack", "skill", "burst"):
                totals[team][action["slot"]] += 1
        for event in turn["events"]:
            team = team_of(event.get("actor"))
            if team:
                totals[team]["damage"] += float(event.get("hp_damage", 0))
                totals[team]["shield_damage"] += float(event.get("shield_absorbed", 0))
                totals[team]["healing"] += float(event.get("restored", 0))
                totals[team]["energy_spent"] += int(event.get("energy_spent", 0))
                totals[team]["energy_gained"] += int(event.get("energy_gained", 0))
                totals[team]["reactions"] += int(event.get("kind") == "reaction")
                totals[team]["frozen_actions"] += int(event.get("kind") == "frozen")
                totals[team]["invalid_actions"] += int(event.get("kind") == "invalid")
            if event.get("kind") == "environment" and event.get("transition") == "start":
                name = event.get("event")
                if name in arena_events:
                    arena_events[name] += 1
            if event.get("kind") == "synergy_change" and event.get("team") in totals:
                totals[event["team"]]["synergy_changes"] += 1

    survival = {}
    ending_energy = {}
    survivors = {}
    for team in ("A", "B"):
        fighters = [fighter for fighter in battle.fighters.values() if fighter.team == team]
        survival[team] = sum(fighter.hp for fighter in fighters) / sum(fighter.max_hp for fighter in fighters)
        ending_energy[team] = sum(fighter.energy for fighter in fighters)
        survivors[team] = [fighter.data["name"] for fighter in fighters if fighter.alive]

    initial_synergy = battle.initial_state["synergy"]
    row = {
        "match_id": f"MATCH-{match_id:06d}", "runner_seed": runner_seed, "battle_seed": battle_seed,
        "team_size": len(team_a), "team_a": "|".join(team_a), "team_b": "|".join(team_b),
        "stat_power_a": powers["A"], "stat_power_b": powers["B"], "weaker_team": weaker,
        "weather": battle.arena.weather, "objective": battle.objective.mode,
        "controller_a": battle.baseline_controllers["A"], "controller_b": battle.baseline_controllers["B"],
        "max_turns": battle.max_turns, "events_enabled": int(battle.arena.events_enabled),
        "reactions_enabled": int(battle.elements.enabled), "synergy_enabled": int(battle.synergy.enabled),
        "winner": result["winner"] or "Draw",
        "upset": int(result["winner"] is not None and result["winner"] == weaker),
        "reason": result["reason"], "turns": result["turns"],
        "objective_score_a": result["scores"]["A"], "objective_score_b": result["scores"]["B"],
        "synergy_score_a": initial_synergy["A"]["active_score"],
        "synergy_score_b": initial_synergy["B"]["active_score"],
        "survivors_a": "|".join(survivors["A"]), "survivors_b": "|".join(survivors["B"]),
        "survival_fraction_a": round(survival["A"], 6), "survival_fraction_b": round(survival["B"], 6),
        "energy_blackouts": arena_events["energy_blackout"],
        "shield_nullifications": arena_events["shield_nullification"],
        "ending_energy_a": ending_energy["A"], "ending_energy_b": ending_energy["B"],
    }
    metric_names = ("damage", "shield_damage", "healing", "energy_spent", "energy_gained",
                    "reactions", "synergy_changes", "frozen_actions", "move", "defend",
                    "normal_attack", "skill", "burst", "invalid_actions")
    output_names = {"move": "moves", "defend": "defends", "normal_attack": "normal_attacks",
                    "skill": "skills", "burst": "bursts"}
    for metric in metric_names:
        output = output_names.get(metric, metric)
        for team in ("A", "B"):
            value = totals[team][metric]
            row[f"{output}_{team.lower()}"] = round(value, 4) if isinstance(value, float) else value
    for component in ("survival", "objective_control", "resource_management", "team_synergy",
                      "damage_contribution", "adaptability", "risk_efficiency"):
        for team in ("A", "B"):
            row[f"reward_{component}_{team.lower()}"] = rewards["teams"][team]["components"][component]
    for team in ("A", "B"):
        row[f"weighted_score_{team.lower()}"] = rewards["teams"][team]["weighted_score"]
        row[f"relative_reward_{team.lower()}"] = rewards["teams"][team]["relative_reward"]
    return {field: row[field] for field in FIELDS}


def sample_scenario(rng, roster_names, requested_size):
    size = rng.randint(1, 4) if requested_size == "random" else int(requested_size)
    chosen = rng.sample(roster_names, size * 2)
    return {
        "team_a": chosen[:size], "team_b": chosen[size:],
        "weather": rng.choice(WEATHERS), "objective": rng.choice(OBJECTIVE_TYPES),
        "controller_a": rng.choice(CONTROLLERS), "controller_b": rng.choice(CONTROLLERS),
        "battle_seed": rng.randrange(2 ** 31),
    }


def run_matches(rules, roster, count, runner_seed, team_size, max_turns, target_score,
                events_enabled, reactions_enabled, synergy_enabled, synergy_path,
                logs_dir=None, progress_every=50):
    rng = random.Random(runner_seed)
    roster_names = sorted(roster)
    maximum_size = min(4, len(roster_names) // 2)
    if maximum_size < 1:
        raise ValueError("At least two configured characters are required.")
    if team_size != "random" and int(team_size) > maximum_size:
        raise ValueError(f"team-size cannot exceed {maximum_size} for this roster.")
    rows = []
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        scenario = sample_scenario(rng, roster_names, team_size)
        battle = Battle(
            rules, roster, scenario["team_a"], scenario["team_b"],
            seed=scenario["battle_seed"], max_turns=max_turns,
            weather=scenario["weather"], events_enabled=events_enabled,
            objective=scenario["objective"], target_score=target_score,
            controller_a=scenario["controller_a"], controller_b=scenario["controller_b"],
            reactions_enabled=reactions_enabled, synergy_enabled=synergy_enabled,
            synergy_path=synergy_path,
        )
        while not battle.finished:
            battle.step()
        rows.append(summarize(battle, index, runner_seed, scenario["battle_seed"],
                              scenario["team_a"], scenario["team_b"]))
        if logs_dir is not None:
            path = logs_dir / f"match_{index:06d}.json"
            full_log = battle.export_log()
            full_log["reward_evaluation"] = calculate_rewards(battle)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(full_log, handle, indent=2)
        if progress_every and (index % progress_every == 0 or index == count):
            print(f"Completed {index}/{count} matches.")
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False,
                                     dir=output_path.parent, prefix=output_path.name + ".",
                                     suffix=".tmp") as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="Prepared or original character CSV")
    parser.add_argument("--abilities", type=Path, default=DEFAULT_ABILITIES)
    parser.add_argument("--synergies", type=Path, default=DEFAULT_SYNERGIES)
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026, help="Seed controlling every sampled scenario")
    parser.add_argument("--team-size", choices=("random", "1", "2", "3", "4"), default="random")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--target-score", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("simulation_results.csv"))
    parser.add_argument("--logs-dir", type=Path, help="Optional directory for full JSON battle logs")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N matches; 0 disables it")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-reactions", action="store_true")
    parser.add_argument("--no-synergy", action="store_true")
    args = parser.parse_args()
    try:
        if args.matches < 1 or args.max_turns < 1 or args.target_score < 1 or args.progress_every < 0:
            raise ValueError("matches, max-turns, and target-score must be positive; progress-every cannot be negative.")
        protected = [Path(__file__), args.abilities, args.synergies,
                     PROJECT_DIR / "battle_engine.py", PROJECT_DIR / "ability_loader.py",
                     PROJECT_DIR / "data_loader.py", PROJECT_DIR / "arena_environment.py",
                     PROJECT_DIR / "arena_objectives.py", PROJECT_DIR / "elemental_system.py",
                     PROJECT_DIR / "synergy_system.py", PROJECT_DIR / "synergies.json",
                     PROJECT_DIR / "reward_calculator.py"]
        if args.csv is not None:
            protected.append(args.csv)
        if any(args.output.resolve() == path.resolve() for path in protected):
            raise ValueError("Choose an output filename that does not overwrite a source or rules file.")
        rules, roster = load_combat_roster(args.csv, args.abilities)
        rows = run_matches(
            rules, roster, args.matches, args.seed, args.team_size, args.max_turns,
            args.target_score, not args.no_events, not args.no_reactions,
            not args.no_synergy, args.synergies, args.logs_dir, args.progress_every,
        )
        write_csv(rows, args.output)
        wins = {name: sum(row["winner"] == name for row in rows) for name in ("A", "B", "Draw")}
        upsets = sum(row["upset"] for row in rows)
        average_turns = sum(row["turns"] for row in rows) / len(rows)
        print(f"Saved {len(rows)} match summaries to: {args.output.resolve()}")
        print(f"Results: A wins {wins['A']}, B wins {wins['B']}, draws {wins['Draw']}.")
        print(f"Statistical upsets: {upsets}. Average turns: {average_turns:.1f}.")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
