"""Fairly compare random, rule-based, and learned arena controllers.

Every controller is evaluated on the same character teams, weather, objective,
event seed, and turn limit. Team B always follows a random legal policy. Results
are written to a CSV so they can be analysed or plotted later.

Example:
    python compare_agents.py --model linear_q_model.json --matches 100
"""

import argparse
import csv
from pathlib import Path
import random

from arena_objectives import OBJECTIVE_TYPES
from rl_environment import (
    ABILITY_SLOTS,
    ABILITY_START,
    DEFEND_ACTION,
    GLOBAL_FIGHTER_SLOTS,
    GRID_SIZE,
    MAX_TEAM_SIZE,
    MOVE_START,
    GenshinArenaEnv,
)
from train_linear_q import LinearQAgent, legal_codes


CONTROLLERS = ("random", "rule_based", "linear_q")
WEATHERS = ("clear", "blizzard", "thunderstorm")
COMPONENTS = (
    "survival",
    "objective_control",
    "resource_management",
    "team_synergy",
    "damage_contribution",
    "adaptability",
    "risk_efficiency",
)


def generate_scenarios(roster, matches, seed, team_size):
    rng = random.Random(f"{seed}:scenarios")
    names = sorted(roster)
    scenarios = []
    for match in range(1, matches + 1):
        size = rng.randint(1, 4) if team_size == "random" else int(team_size)
        selected = rng.sample(names, size * 2)
        scenarios.append({
            "match": match,
            "team_size": size,
            "team_a": selected[:size],
            "team_b": selected[size:],
            "weather": rng.choice(WEATHERS),
            "objective": rng.choice(OBJECTIVE_TYPES),
            "battle_seed": rng.randrange(2 ** 31),
        })
    return scenarios


def random_actions(observation, team, rng):
    return [rng.choice(legal_codes(mask)) for mask in observation["action_masks"][team]]


def encode_rule_action(env, action, mask):
    """Translate one engine action dictionary into the fixed RL action code."""
    kind = action.get("kind")
    if kind == "defend":
        code = DEFEND_ACTION
    elif kind == "move":
        x, y = action["destination"]
        code = MOVE_START + x * GRID_SIZE + y
    elif kind == "ability":
        slot_index = ABILITY_SLOTS.index(action["slot"])
        target_index = env.global_slots.index(action["target"])
        code = ABILITY_START + slot_index * GLOBAL_FIGHTER_SLOTS + target_index
    else:
        return DEFEND_ACTION
    return code if 0 <= code < len(mask) and mask[code] else DEFEND_ACTION


def rule_based_actions(env, observation, team="A"):
    """Ask the battle engine's inspectable objective-aware controller."""
    reserved = set()
    masks = observation["action_masks"][team]
    codes = []
    for slot, uid in enumerate(env.team_slots[team]):
        fighter = env.battle.fighters.get(uid) if uid else None
        if fighter is None or not fighter.alive:
            codes.append(DEFEND_ACTION)
            continue
        action = env.battle.choose_action(fighter, reserved)
        codes.append(encode_rule_action(env, action, masks[slot]))
    return codes


def run_match(env, scenario, controller, learned_agent, seed):
    options = {key: scenario[key] for key in
               ("team_size", "team_a", "team_b", "weather", "objective", "battle_seed")}
    observation, _ = env.reset(options=options)
    opponent_rng = random.Random(f"{seed}:{scenario['match']}:opponent")
    controller_rng = random.Random(f"{seed}:{scenario['match']}:{controller}")
    total_reward = 0.0
    done = False
    while not done:
        if controller == "random":
            action_a = random_actions(observation, "A", controller_rng)
        elif controller == "rule_based":
            action_a = rule_based_actions(env, observation)
        else:
            learned_agent.rng = controller_rng
            action_a = learned_agent.joint_action(observation, "A", training=False)
        action_b = random_actions(observation, "B", opponent_rng)
        observation, rewards, terminated, truncated, info = env.step({"A": action_a, "B": action_b})
        total_reward += rewards["A"]
        done = terminated or truncated

    result = info["result"]
    evaluation = info["evaluation"]["teams"]["A"]
    components = evaluation["components"]
    row = {
        "match": scenario["match"],
        "controller": controller,
        "team_size": scenario["team_size"],
        "team_a": "|".join(scenario["team_a"]),
        "team_b": "|".join(scenario["team_b"]),
        "weather": scenario["weather"],
        "objective": scenario["objective"],
        "battle_seed": scenario["battle_seed"],
        "winner": result["winner"] or "Draw",
        "team_a_win": int(result["winner"] == "A"),
        "draw": int(result["winner"] is None),
        "turns": result["turns"],
        "episode_reward": round(total_reward, 6),
        "weighted_score": evaluation["weighted_score"],
        "objective_score_a": result["scores"]["A"],
        "objective_score_b": result["scores"]["B"],
        "survivors_a": len(result["survivors"]["A"]),
        "survivors_b": len(result["survivors"]["B"]),
    }
    row.update({f"reward_{name}": components[name] for name in COMPONENTS})
    return row


def summarize(rows, matches):
    print("\nController comparison")
    print("-" * 86)
    print(f"{'Controller':<14} {'A wins':>8} {'B wins':>8} {'Draws':>8} "
          f"{'Win rate':>10} {'Avg reward':>12} {'Avg score':>11}")
    print("-" * 86)
    summaries = {}
    for controller in CONTROLLERS:
        selected = [row for row in rows if row["controller"] == controller]
        a_wins = sum(row["team_a_win"] for row in selected)
        draws = sum(row["draw"] for row in selected)
        b_wins = matches - a_wins - draws
        average_reward = sum(row["episode_reward"] for row in selected) / matches
        average_score = sum(row["weighted_score"] for row in selected) / matches
        summaries[controller] = (a_wins, b_wins, draws, average_reward, average_score)
        print(f"{controller:<14} {a_wins:>8} {b_wins:>8} {draws:>8} "
              f"{a_wins / matches:>9.1%} {average_reward:>12.4f} {average_score:>11.4f}")
    print("-" * 86)
    learned = summaries["linear_q"]
    random_result = summaries["random"]
    print(f"Linear Q versus random: win-rate change "
          f"{(learned[0] - random_result[0]) / matches:+.1%}, "
          f"average-reward change {learned[3] - random_result[3]:+.4f}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--abilities", type=Path, default=Path(__file__).with_name("abilities.json"))
    parser.add_argument("--synergies", type=Path, default=Path(__file__).with_name("synergies.json"))
    parser.add_argument("--model", type=Path, default=Path("linear_q_model.json"))
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--team-size", choices=("random", "1", "2", "3", "4"), default="random")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("agent_comparison.csv"))
    args = parser.parse_args()
    try:
        if args.matches < 1:
            raise ValueError("matches must be positive.")
        learned_agent = LinearQAgent.load(args.model, args.seed)
        team_size = args.team_size if args.team_size == "random" else int(args.team_size)
        env = GenshinArenaEnv(
            csv_path=args.csv, abilities_path=args.abilities, synergies_path=args.synergies,
            seed=args.seed, team_size=team_size, max_turns=args.max_turns,
        )
        scenarios = generate_scenarios(env.roster, args.matches, args.seed, team_size)
        rows = []
        for controller in CONTROLLERS:
            for scenario in scenarios:
                rows.append(run_match(env, scenario, controller, learned_agent, args.seed))
            print(f"Completed {controller}: {args.matches} matches.")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summarize(rows, args.matches)
        print(f"Detailed report saved: {args.output.resolve()}")
        print("Use at least 100 matches before drawing conclusions from win rate.")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
