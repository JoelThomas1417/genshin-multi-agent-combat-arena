"""Train a MAPPO-style neural controller for the Genshin combat arena.

The four Team A characters share one actor network. A centralized critic sees
the complete Team A observation vector and estimates the team's future return.
Legal-action masks prevent the policy from selecting impossible actions.

Install once:
    python -m pip install -r requirements_mappo.txt

Train and save a checkpoint:
    python train_mappo.py --episodes 1000 --model mappo_model.pt

Train with an easy-to-hard curriculum:
    python train_mappo.py --curriculum --episodes 1500 --model mappo_curriculum.pt

Continue an existing checkpoint:
    python train_mappo.py --resume --episodes 500 --model mappo_model.pt

Evaluate a saved checkpoint:
    python train_mappo.py --evaluate-only --model mappo_model.pt
"""

import argparse
import csv
from pathlib import Path
import random
import sys

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError as error:
    print("MAPPO requires NumPy and PyTorch.", file=sys.stderr)
    print("Install them with: python -m pip install -r requirements_mappo.txt", file=sys.stderr)
    raise SystemExit(1) from error

from compare_agents import rule_based_actions
from rl_environment import ACTION_SPACE_SIZE, MAX_TEAM_SIZE, GenshinArenaEnv


CHECKPOINT_VERSION = 1


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, ACTION_SPACE_SIZE),
        )

    def forward(self, state):
        return self.network(state)


class CentralCritic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)


def actor_inputs(observation, team="A"):
    """Add a slot identity to the shared global observation."""
    global_vector = np.asarray(observation["vectors"][team], dtype=np.float32)
    rows = []
    for slot in range(MAX_TEAM_SIZE):
        identity = np.zeros(MAX_TEAM_SIZE, dtype=np.float32)
        identity[slot] = 1.0
        rows.append(np.concatenate((global_vector, identity)))
    return np.stack(rows)


def masked_distribution(actor, inputs, masks, device):
    tensor_inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    tensor_masks = torch.as_tensor(masks, dtype=torch.bool, device=device)
    logits = actor(tensor_inputs)
    logits = logits.masked_fill(~tensor_masks, -1e9)
    return Categorical(logits=logits)


def select_actions(actor, observation, device, deterministic=False):
    inputs = actor_inputs(observation)
    masks = np.asarray(observation["action_masks"]["A"], dtype=np.bool_)
    with torch.no_grad():
        distribution = masked_distribution(actor, inputs, masks, device)
        actions = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)
    return actions.cpu().tolist(), log_probs.cpu().numpy(), inputs, masks


def random_actions(observation, team, rng):
    actions = []
    for mask in observation["action_masks"][team]:
        legal = [code for code, allowed in enumerate(mask) if allowed]
        actions.append(rng.choice(legal))
    return actions


def opponent_actions(env, observation, mode, rng):
    if mode == "mixed":
        mode = "rule_based" if rng.random() < 0.5 else "random"
    if mode == "rule_based":
        return rule_based_actions(env, observation, "B")
    return random_actions(observation, "B", rng)


def calculate_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for step in reversed(range(len(rewards))):
        continuation = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * next_value * continuation - values[step]
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[step] = gae
        next_value = values[step]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def collect_episode(env, actor, critic, device, opponent, opponent_rng):
    observation, _ = env.reset()
    trajectory = {key: [] for key in
                  ("global_states", "actor_inputs", "masks", "actions",
                   "old_log_probs", "rewards", "values", "dones")}
    done = False
    while not done:
        actions_a, log_probs, inputs, masks = select_actions(actor, observation, device)
        global_state = np.asarray(observation["vectors"]["A"], dtype=np.float32)
        with torch.no_grad():
            value = critic(torch.as_tensor(global_state, device=device).unsqueeze(0)).item()
        actions_b = opponent_actions(env, observation, opponent, opponent_rng)
        next_observation, rewards, terminated, truncated, _ = env.step({"A": actions_a, "B": actions_b})
        done = terminated or truncated
        trajectory["global_states"].append(global_state)
        trajectory["actor_inputs"].append(inputs)
        trajectory["masks"].append(masks)
        trajectory["actions"].append(np.asarray(actions_a, dtype=np.int64))
        trajectory["old_log_probs"].append(log_probs.astype(np.float32))
        trajectory["rewards"].append(float(rewards["A"]))
        trajectory["values"].append(float(value))
        trajectory["dones"].append(done)
        observation = next_observation
    return trajectory, env.battle.result()


def combine_trajectories(trajectories, gamma, gae_lambda):
    combined = {key: [] for key in
                ("global_states", "actor_inputs", "masks", "actions",
                 "old_log_probs", "advantages", "returns")}
    episode_returns = []
    for trajectory in trajectories:
        advantages, returns = calculate_gae(
            trajectory["rewards"], trajectory["values"], trajectory["dones"],
            gamma, gae_lambda,
        )
        for key in ("global_states", "actor_inputs", "masks", "actions", "old_log_probs"):
            combined[key].extend(trajectory[key])
        combined["advantages"].extend(advantages)
        combined["returns"].extend(returns)
        episode_returns.append(sum(trajectory["rewards"]))
    return combined, episode_returns


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, batch, args, device):
    global_states = torch.as_tensor(np.asarray(batch["global_states"]), dtype=torch.float32, device=device)
    returns = torch.as_tensor(np.asarray(batch["returns"]), dtype=torch.float32, device=device)
    advantages = torch.as_tensor(np.asarray(batch["advantages"]), dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    inputs = torch.as_tensor(np.asarray(batch["actor_inputs"]), dtype=torch.float32, device=device)
    masks = torch.as_tensor(np.asarray(batch["masks"]), dtype=torch.bool, device=device)
    actions = torch.as_tensor(np.asarray(batch["actions"]), dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(np.asarray(batch["old_log_probs"]), dtype=torch.float32, device=device)
    actor_advantages = advantages.unsqueeze(1).expand(-1, MAX_TEAM_SIZE)

    policy_losses, value_losses, entropies = [], [], []
    for _ in range(args.ppo_epochs):
        distribution = masked_distribution(actor, inputs.reshape(-1, inputs.shape[-1]),
                                           masks.reshape(-1, ACTION_SPACE_SIZE), device)
        new_log_probs = distribution.log_prob(actions.reshape(-1)).reshape_as(actions)
        entropy = distribution.entropy().mean()
        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * actor_advantages
        clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * actor_advantages
        policy_loss = -torch.min(unclipped, clipped).mean() - args.entropy_coef * entropy

        actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_optimizer.step()

        predicted_values = critic(global_states)
        value_loss = 0.5 * (predicted_values - returns).pow(2).mean()
        critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        critic_optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropies.append(entropy.item())
    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)


def evaluate(env, actor, device, episodes, opponent, seed):
    wins = {"A": 0, "B": 0, "Draw": 0}
    returns = []
    rng = random.Random(f"{seed}:evaluation:{opponent}")
    actor.eval()
    for _ in range(episodes):
        observation, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            actions_a, _, _, _ = select_actions(actor, observation, device, deterministic=True)
            actions_b = opponent_actions(env, observation, opponent, rng)
            observation, rewards, terminated, truncated, info = env.step({"A": actions_a, "B": actions_b})
            total_reward += rewards["A"]
            done = terminated or truncated
        wins[info["result"]["winner"] or "Draw"] += 1
        returns.append(total_reward)
    actor.train()
    return wins, float(np.mean(returns))


def save_checkpoint(path, actor, critic, args, observation_size, episodes_trained=0,
                    actor_optimizer=None, critic_optimizer=None):
    torch.save({
        "checkpoint_version": CHECKPOINT_VERSION,
        "algorithm": "mappo_style_shared_actor_central_critic",
        "observation_size": observation_size,
        "actor_input_size": observation_size + MAX_TEAM_SIZE,
        "action_space_size": ACTION_SPACE_SIZE,
        "hidden_size": actor.network[0].out_features,
        "episodes_trained": episodes_trained,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict() if actor_optimizer else None,
        "critic_optimizer_state_dict": critic_optimizer.state_dict() if critic_optimizer else None,
        "training_arguments": vars(args),
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported MAPPO checkpoint version.")
    actor = Actor(checkpoint["actor_input_size"], checkpoint["hidden_size"]).to(device)
    critic = CentralCritic(checkpoint["observation_size"], checkpoint["hidden_size"]).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    return actor, critic, checkpoint["observation_size"], checkpoint


def curriculum_opponent(completed, total):
    """Progress from basic exploration to increasingly strong opposition."""
    progress = completed / max(1, total)
    if progress < 0.40:
        return "random"
    if progress < 0.80:
        return "mixed"
    return "rule_based"


def append_history(path, row):
    """Append one PPO update summary without deleting earlier resumed runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_env(args, seed):
    team_size = args.team_size if args.team_size == "random" else int(args.team_size)
    return GenshinArenaEnv(
        csv_path=args.csv, abilities_path=args.abilities, synergies_path=args.synergies,
        seed=seed, team_size=team_size, max_turns=args.max_turns,
    )


def print_evaluation(label, wins, average_return, episodes):
    print(f"{label}: A wins {wins['A']}, B wins {wins['B']}, draws {wins['Draw']} | "
          f"A win rate {wins['A'] / episodes:.1%} | average reward {average_return:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--abilities", type=Path, default=Path(__file__).with_name("abilities.json"))
    parser.add_argument("--synergies", type=Path, default=Path(__file__).with_name("synergies.json"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episodes-per-update", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--team-size", choices=("random", "1", "2", "3", "4"), default="random")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--opponent", choices=("random", "rule_based", "mixed"), default="mixed")
    parser.add_argument("--curriculum", action="store_true",
                        help="Train 40%% random, 40%% mixed, then 20%% rule-based")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--model", type=Path, default=Path("mappo_model.pt"))
    parser.add_argument("--history", type=Path, default=Path("mappo_training_history.csv"))
    parser.add_argument("--resume", action="store_true",
                        help="Continue training the checkpoint in --model")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    try:
        if min(args.episodes, args.episodes_per_update, args.evaluation_episodes) < 1:
            raise ValueError("Episode counts must be positive.")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        device = torch.device("cuda" if args.device == "cuda" or
                              (args.device == "auto" and torch.cuda.is_available()) else "cpu")
        set_seeds(args.seed)
        env = make_env(args, args.seed)
        probe, _ = env.reset()
        observation_size = len(probe["vectors"]["A"])

        if args.evaluate_only:
            actor, _, saved_size, _ = load_checkpoint(args.model, device)
            if saved_size != observation_size:
                raise ValueError("Checkpoint observation size does not match this environment.")
            for opponent in ("random", "rule_based"):
                wins, average_return = evaluate(env, actor, device, args.evaluation_episodes,
                                                opponent, args.seed + 2_000_000)
                print_evaluation(f"Against {opponent}", wins, average_return,
                                 args.evaluation_episodes)
            return

        previous_episodes = 0
        checkpoint = None
        if args.resume:
            actor, critic, saved_size, checkpoint = load_checkpoint(args.model, device)
            if saved_size != observation_size:
                raise ValueError("Checkpoint observation size does not match this environment.")
            previous_episodes = int(checkpoint.get(
                "episodes_trained", checkpoint.get("training_arguments", {}).get("episodes", 0)))
            print(f"Resuming {args.model} after {previous_episodes} recorded episodes.")
        else:
            actor = Actor(observation_size + MAX_TEAM_SIZE, args.hidden_size).to(device)
            critic = CentralCritic(observation_size, args.hidden_size).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
        if checkpoint:
            if checkpoint.get("actor_optimizer_state_dict"):
                actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            if checkpoint.get("critic_optimizer_state_dict"):
                critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        opponent_rng = random.Random(f"{args.seed}:training-opponent")
        completed = 0
        while completed < args.episodes:
            count = min(args.episodes_per_update, args.episodes - completed)
            training_opponent = (curriculum_opponent(completed, args.episodes)
                                 if args.curriculum else args.opponent)
            trajectories, winners = [], []
            for _ in range(count):
                trajectory, result = collect_episode(
                    env, actor, critic, device, training_opponent, opponent_rng)
                trajectories.append(trajectory)
                winners.append(result["winner"] or "Draw")
            batch, episode_returns = combine_trajectories(
                trajectories, args.gamma, args.gae_lambda)
            policy_loss, value_loss, entropy = ppo_update(
                actor, critic, actor_optimizer, critic_optimizer, batch, args, device)
            completed += count
            total_trained = previous_episodes + completed
            average_reward = float(np.mean(episode_returns))
            append_history(args.history, {
                "total_episodes_trained": total_trained,
                "run_episode": completed,
                "run_target": args.episodes,
                "opponent_stage": training_opponent,
                "batch_episodes": count,
                "a_wins": winners.count("A"),
                "b_wins": winners.count("B"),
                "draws": winners.count("Draw"),
                "average_reward": round(average_reward, 6),
                "policy_loss": round(float(policy_loss), 6),
                "value_loss": round(float(value_loss), 6),
                "entropy": round(float(entropy), 6),
            })
            save_checkpoint(args.model, actor, critic, args, observation_size,
                            total_trained, actor_optimizer, critic_optimizer)
            print(f"Training {completed}/{args.episodes} [{training_opponent}]: "
                  f"A wins {winners.count('A')}/{count}, "
                  f"reward {average_reward:.4f}, policy loss {policy_loss:.4f}, "
                  f"value loss {value_loss:.4f}, entropy {entropy:.4f}")

        print(f"Saved MAPPO checkpoint: {args.model.resolve()}")
        print(f"Training history: {args.history.resolve()}")
        for opponent in ("random", "rule_based"):
            wins, average_return = evaluate(env, actor, device, args.evaluation_episodes,
                                            opponent, args.seed + 2_000_000)
            print_evaluation(f"Against {opponent}", wins, average_return,
                             args.evaluation_episodes)
        print(f"Device used: {device}; observation values: {observation_size}; actions per character: {ACTION_SPACE_SIZE}.")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
