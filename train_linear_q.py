"""Train the arena's first reinforcement-learning combat controller.

This dependency-free baseline trains Team A against a random legal Team B.
It uses linear Q-learning, respects the environment's action masks, reports a
before/after evaluation, and saves the learned weights as JSON.

Examples:
    python train_linear_q.py --episodes 500
    python train_linear_q.py --episodes 1000 --team-size 2 --model linear_q_model.json
    python train_linear_q.py --evaluate-only --model linear_q_model.json

This is intentionally a small baseline, not the final MAPPO/PPO controller.
Its purpose is to verify that actions can improve through reward feedback.
"""

import argparse
import json
import math
from pathlib import Path
import random

from rl_environment import ACTION_SPACE_SIZE, MAX_TEAM_SIZE, GenshinArenaEnv


MODEL_VERSION = 1
TEAM = "A"


def legal_codes(mask):
    """Return the legal integer action codes in one action mask."""
    return [code for code, allowed in enumerate(mask) if allowed]


def _mean(values, default=0.0):
    return sum(values) / len(values) if values else default


def agent_features(observation, team, slot, max_turns):
    """Convert the readable observation into small normalized features.

    The policy receives current public information only. It cannot see the
    hidden future event draw.
    """
    state = observation["state"]
    enemy = "B" if team == "A" else "A"
    fighters = state["fighters"]
    own = [fighter for fighter in fighters.values() if fighter["team"] == team]
    foes = [fighter for fighter in fighters.values() if fighter["team"] == enemy]
    alive_own = [fighter for fighter in own if fighter["alive"]]
    alive_foes = [fighter for fighter in foes if fighter["alive"]]
    ordered_own = list(own)
    actor = ordered_own[slot] if slot < len(ordered_own) else None
    arena = state["arena"]
    event = arena.get("active_event")
    event_name = event.get("name") if event else None
    objective = state["objective"]
    target_score = max(1, objective.get("target_score") or 10)

    own_hp = _mean([f["hp"] / f["max_hp"] for f in alive_own])
    foe_hp = _mean([f["hp"] / f["max_hp"] for f in alive_foes])
    own_energy = _mean([f["energy"] / 100.0 for f in alive_own])
    own_shield = _mean([min(1.0, f["shield"] / f["max_hp"]) for f in alive_own])

    actor_alive = float(bool(actor and actor["alive"]))
    actor_hp = actor["hp"] / actor["max_hp"] if actor else 0.0
    actor_energy = actor["energy"] / 100.0 if actor else 0.0
    actor_shield = min(1.0, actor["shield"] / actor["max_hp"]) if actor else 0.0
    actor_x = actor["position"][0] / 7.0 if actor else 0.0
    actor_y = actor["position"][1] / 7.0 if actor else 0.0
    actor_frozen = float(bool(actor and actor["elemental_status"]["actions_to_skip"]))
    if actor and alive_foes:
        distances = [abs(actor["position"][0] - foe["position"][0])
                     + abs(actor["position"][1] - foe["position"][1]) for foe in alive_foes]
        nearest_enemy = min(distances) / 14.0
    else:
        nearest_enemy = 1.0

    zone_cells = {tuple(cell) for cell in objective.get("zone_cells", [])}
    actor_in_zone = float(bool(actor and tuple(actor["position"]) in zone_cells))
    enemy_in_zone = float(any(tuple(foe["position"]) in zone_cells for foe in alive_foes))
    scores = objective.get("scores", {"A": 0, "B": 0})

    return [
        1.0,
        min(1.0, state["turn"] / max_turns),
        float(objective["type"] == "control_zone"),
        min(1.0, scores[team] / target_score),
        min(1.0, scores[enemy] / target_score),
        float(arena["weather"] == "clear"),
        float(arena["weather"] == "blizzard"),
        float(arena["weather"] == "thunderstorm"),
        float(event_name is None),
        float(event_name == "energy_blackout"),
        float(event_name == "shield_nullification"),
        len(alive_own) / MAX_TEAM_SIZE,
        len(alive_foes) / MAX_TEAM_SIZE,
        own_hp,
        foe_hp,
        own_energy,
        own_shield,
        actor_alive,
        actor_hp,
        actor_energy,
        actor_shield,
        actor_x,
        actor_y,
        actor_frozen,
        nearest_enemy,
        actor_in_zone,
        enemy_in_zone,
        min(1.0, state["synergy"][team]["active_score"] / 100.0),
    ]


class LinearQAgent:
    """Four-slot cooperative linear Q-learning policy."""

    def __init__(self, feature_count, seed=2026, learning_rate=0.05,
                 gamma=0.97, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, max_turns=100):
        self.feature_count = feature_count
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_turns = max_turns
        self.rng = random.Random(seed)
        self.weights = [
            [[0.0] * feature_count for _ in range(ACTION_SPACE_SIZE)]
            for _ in range(MAX_TEAM_SIZE)
        ]
        self.visits = [[0] * ACTION_SPACE_SIZE for _ in range(MAX_TEAM_SIZE)]

    def q_value(self, slot, action, features):
        return sum(weight * value for weight, value in zip(self.weights[slot][action], features))

    def choose(self, observation, team, slot, training=True):
        mask = observation["action_masks"][team][slot]
        choices = legal_codes(mask)
        if not choices:
            return 0
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(choices)
        features = agent_features(observation, team, slot, self.max_turns)
        scored = [(self.q_value(slot, action, features), action) for action in choices]
        best_value = max(score for score, _ in scored)
        best = [action for score, action in scored if abs(score - best_value) < 1e-12]
        return self.rng.choice(best)

    def joint_action(self, observation, team=TEAM, training=True):
        return [self.choose(observation, team, slot, training) for slot in range(MAX_TEAM_SIZE)]

    def update(self, observation, actions, reward, next_observation, done, team=TEAM,
               max_turns=None):
        max_turns = max_turns or self.max_turns
        for slot, action in enumerate(actions):
            features = agent_features(observation, team, slot, max_turns)
            current = self.q_value(slot, action, features)
            if done:
                future = 0.0
            else:
                next_features = agent_features(next_observation, team, slot, max_turns)
                next_actions = legal_codes(next_observation["action_masks"][team][slot])
                future = max((self.q_value(slot, candidate, next_features)
                              for candidate in next_actions), default=0.0)
            target = reward + self.gamma * future
            td_error = max(-1.0, min(1.0, target - current))
            self.visits[slot][action] += 1
            rate = self.learning_rate / math.sqrt(1.0 + self.visits[slot][action] / 25.0)
            row = self.weights[slot][action]
            for index, value in enumerate(features):
                row[index] = max(-5.0, min(5.0, row[index] + rate * td_error * value))

    def finish_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        payload = {
            "model_version": MODEL_VERSION,
            "algorithm": "linear_q_learning",
            "trained_team": TEAM,
            "feature_count": self.feature_count,
            "action_space_size": ACTION_SPACE_SIZE,
            "maximum_team_slots": MAX_TEAM_SIZE,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "max_turns": self.max_turns,
            },
            "weights": self.weights,
            "visits": self.visits,
        }
        Path(path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path, seed=2026):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError("Unsupported model version.")
        hp = payload["hyperparameters"]
        agent = cls(payload["feature_count"], seed, hp["learning_rate"], hp["gamma"],
                    hp["epsilon"], hp["epsilon_min"], hp["epsilon_decay"],
                    hp.get("max_turns", 100))
        agent.weights = payload["weights"]
        agent.visits = payload["visits"]
        return agent


def random_joint_action(observation, team, rng):
    return [rng.choice(legal_codes(mask)) for mask in observation["action_masks"][team]]


def make_env(args, seed):
    team_size = args.team_size if args.team_size == "random" else int(args.team_size)
    return GenshinArenaEnv(
        csv_path=args.csv, abilities_path=args.abilities, synergies_path=args.synergies,
        seed=seed, team_size=team_size, max_turns=args.max_turns,
    )


def play_episode(env, agent, opponent_rng, training):
    observation, _ = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action_a = agent.joint_action(observation, TEAM, training=training)
        action_b = random_joint_action(observation, "B", opponent_rng)
        next_observation, rewards, terminated, truncated, info = env.step({"A": action_a, "B": action_b})
        done = terminated or truncated
        if training:
            agent.update(observation, action_a, rewards["A"], next_observation, done,
                         TEAM, env.max_turns)
        observation = next_observation
        total_reward += rewards["A"]
        steps += 1
    if training:
        agent.finish_episode()
    winner = info["result"]["winner"] or "Draw"
    return winner, total_reward, steps


def evaluate(args, agent, seed, episodes):
    env = make_env(args, seed)
    opponent_rng = random.Random(f"{seed}:evaluation-opponent")
    # A fixed policy RNG makes repeated evaluations comparable.
    agent.rng = random.Random(f"{seed}:evaluation-policy")
    results = {"A": 0, "B": 0, "Draw": 0}
    returns = []
    turns = []
    for _ in range(episodes):
        winner, episode_return, episode_turns = play_episode(env, agent, opponent_rng, False)
        results[winner] += 1
        returns.append(episode_return)
        turns.append(episode_turns)
    return {
        "wins": results,
        "win_rate": results["A"] / episodes,
        "non_loss_rate": (results["A"] + results["Draw"]) / episodes,
        "average_return": sum(returns) / episodes,
        "average_turns": sum(turns) / episodes,
    }


def print_evaluation(label, result):
    wins = result["wins"]
    print(f"{label}: A wins {wins['A']}, B wins {wins['B']}, draws {wins['Draw']} | "
          f"A win rate {result['win_rate']:.1%} | average reward {result['average_return']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--abilities", type=Path, default=Path(__file__).with_name("abilities.json"))
    parser.add_argument("--synergies", type=Path, default=Path(__file__).with_name("synergies.json"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--team-size", choices=("random", "1", "2", "3", "4"), default="random")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--model", type=Path, default=Path("linear_q_model.json"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.episodes < 1 or args.evaluation_episodes < 1:
            raise ValueError("Episode counts must be positive.")
        probe_env = make_env(args, args.seed)
        probe_observation, _ = probe_env.reset()
        feature_count = len(agent_features(probe_observation, TEAM, 0, args.max_turns))

        if args.evaluate_only:
            agent = LinearQAgent.load(args.model, args.seed)
            result = evaluate(args, agent, args.seed + 2_000_000, args.evaluation_episodes)
            print_evaluation("Saved model evaluation", result)
            return

        agent = LinearQAgent(feature_count, args.seed, max_turns=args.max_turns)
        # Untrained epsilon=1 policy is the random baseline.
        baseline = evaluate(args, agent, args.seed + 2_000_000, args.evaluation_episodes)
        print_evaluation("Before training", baseline)

        env = make_env(args, args.seed + 1_000_000)
        opponent_rng = random.Random(f"{args.seed}:training-opponent")
        report_every = max(1, args.episodes // 10)
        recent_returns = []
        recent_wins = 0
        for episode in range(1, args.episodes + 1):
            winner, episode_return, _ = play_episode(env, agent, opponent_rng, True)
            recent_returns.append(episode_return)
            recent_wins += int(winner == "A")
            if episode % report_every == 0 or episode == args.episodes:
                count = len(recent_returns)
                print(f"Training {episode}/{args.episodes}: epsilon {agent.epsilon:.3f}, "
                      f"recent A win rate {recent_wins / count:.1%}, "
                      f"recent reward {sum(recent_returns) / count:.4f}")
                recent_returns.clear()
                recent_wins = 0

        agent.save(args.model)
        trained = evaluate(args, agent, args.seed + 2_000_000, args.evaluation_episodes)
        print_evaluation("After training", trained)
        print(f"Saved learned model: {args.model.resolve()}")
        print("The model learned from the seven-part strategic reward; it did not use future event outcomes.")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
