"""Gym-like multi-agent wrapper around the Genshin combat arena.

No third-party package is required. The API follows the familiar pattern:

    env = GenshinArenaEnv(seed=2026)
    observation, info = env.reset()
    observation, rewards, terminated, truncated, info = env.step({
        "A": [action_code, ...],
        "B": [action_code, ...],
    })

Each team always has four agent slots; unused or knocked-out slots use no-op
code 0. Every active character has the same fixed discrete action space:

    0       defend / no-op
    1..64   move to one of the 64 grid cells (x * 8 + y)
    65..88  use one of 3 ability slots on one of 8 global fighter slots

The included action mask identifies legal codes. The environment exposes both
the readable battle dictionary and a fixed-length numeric vector per team.
Rewards are dense changes in the seven-part relative score. Across an episode,
they measure improvement over the reset state; Team B's reward is always the
negative of Team A's reward.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import random

from ability_loader import DEFAULT_ABILITIES, load_combat_roster
from arena_objectives import OBJECTIVE_TYPES
from battle_engine import Battle, GRID_SIZE
from reward_calculator import calculate_rewards
from synergy_system import DEFAULT_SYNERGIES


MAX_TEAM_SIZE = 4
GLOBAL_FIGHTER_SLOTS = 8
ABILITY_SLOTS = ("normal_attack", "skill", "burst")
ELEMENTS = ("Anemo", "Cryo", "Dendro", "Electro", "Geo", "Hydro", "Pyro")
WEATHERS = ("clear", "blizzard", "thunderstorm")
EVENTS = (None, "energy_blackout", "shield_nullification")

DEFEND_ACTION = 0
MOVE_START = 1
MOVE_END = MOVE_START + GRID_SIZE * GRID_SIZE
ABILITY_START = MOVE_END
ACTION_SPACE_SIZE = ABILITY_START + len(ABILITY_SLOTS) * GLOBAL_FIGHTER_SLOTS


class GenshinArenaEnv:
    """A simultaneous two-team environment with four fixed slots per team."""

    def __init__(self, csv_path=None, abilities_path=DEFAULT_ABILITIES,
                 synergies_path=DEFAULT_SYNERGIES, seed=2026, team_size="random",
                 max_turns=100, target_score=10, events_enabled=True,
                 reactions_enabled=True, synergy_enabled=True):
        if team_size not in ("random", 1, 2, 3, 4):
            raise ValueError("team_size must be random or a whole number from 1 to 4.")
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive whole number.")
        if type(target_score) is not int or target_score < 1:
            raise ValueError("target_score must be a positive whole number.")
        if any(type(value) is not bool for value in (events_enabled, reactions_enabled, synergy_enabled)):
            raise ValueError("Feature switches must be true or false.")
        self.rules, self.roster = load_combat_roster(csv_path, abilities_path)
        self.synergies_path = Path(synergies_path)
        self.seed = seed
        self._rng = random.Random(seed)
        self.team_size = team_size
        self.max_turns = max_turns
        self.target_score = target_score
        self.events_enabled = events_enabled
        self.reactions_enabled = reactions_enabled
        self.synergy_enabled = synergy_enabled
        self.battle = None
        self.team_slots = {"A": [None] * MAX_TEAM_SIZE, "B": [None] * MAX_TEAM_SIZE}
        self.global_slots = [None] * GLOBAL_FIGHTER_SLOTS
        self._previous_relative = {"A": 0.0, "B": 0.0}

    def _sample_teams(self, size):
        chosen = self._rng.sample(sorted(self.roster), size * 2)
        return chosen[:size], chosen[size:]

    def reset(self, seed=None, options=None):
        """Start a match and expose the decision state for turn one."""
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        options = dict(options or {})
        team_a = options.get("team_a")
        team_b = options.get("team_b")
        if (team_a is None) != (team_b is None):
            raise ValueError("Provide both team_a and team_b, or neither.")
        if team_a is None:
            size = options.get("team_size", self.team_size)
            size = self._rng.randint(1, 4) if size == "random" else int(size)
            team_a, team_b = self._sample_teams(size)
        else:
            team_a, team_b = list(team_a), list(team_b)
            size = len(team_a)
            if not 1 <= size <= 4 or len(team_b) != size:
                raise ValueError("Provided teams must have equal sizes from one to four.")
            if "team_size" in options and int(options["team_size"]) != size:
                raise ValueError("Provided teams must match options.team_size.")
        weather = options.get("weather", self._rng.choice(WEATHERS))
        objective = options.get("objective", self._rng.choice(OBJECTIVE_TYPES))
        battle_seed = options.get("battle_seed", self._rng.randrange(2 ** 31))
        self.battle = Battle(
            self.rules, self.roster, team_a, team_b, seed=battle_seed,
            max_turns=self.max_turns, weather=weather,
            events_enabled=self.events_enabled, objective=objective,
            target_score=self.target_score, reactions_enabled=self.reactions_enabled,
            synergy_enabled=self.synergy_enabled, synergy_path=self.synergies_path,
        )
        self.team_slots = {
            team: [f"{team}:{name}" for name in names] + [None] * (MAX_TEAM_SIZE - len(names))
            for team, names in (("A", team_a), ("B", team_b))
        }
        self.global_slots = self.team_slots["A"] + self.team_slots["B"]
        self.battle.begin_turn()
        evaluation = calculate_rewards(self.battle)
        self._previous_relative = {team: evaluation["teams"][team]["relative_reward"] for team in ("A", "B")}
        info = {
            "seed": self.seed, "battle_seed": battle_seed,
            "team_a": list(team_a), "team_b": list(team_b),
            "weather": self.battle.arena.weather, "objective": objective,
            "action_space_size": ACTION_SPACE_SIZE,
            "action_encoding": self.action_encoding(),
            "initial_evaluation": deepcopy(evaluation),
        }
        return self.observation(), info

    def action_encoding(self):
        return {
            "defend": DEFEND_ACTION,
            "move_codes": [MOVE_START, MOVE_END - 1],
            "ability_codes": [ABILITY_START, ACTION_SPACE_SIZE - 1],
            "ability_slots": list(ABILITY_SLOTS),
            "global_fighter_slots": list(self.global_slots),
        }

    def _ability_code(self, slot_index, target_index):
        return ABILITY_START + slot_index * GLOBAL_FIGHTER_SLOTS + target_index

    def action_masks(self):
        """Return [team][agent slot][action code] booleans."""
        if self.battle is None:
            raise ValueError("Call reset() before requesting action masks.")
        masks = {team: [] for team in ("A", "B")}
        for team in ("A", "B"):
            for uid in self.team_slots[team]:
                mask = [False] * ACTION_SPACE_SIZE
                mask[DEFEND_ACTION] = True
                fighter = self.battle.fighters.get(uid) if uid else None
                if fighter is not None and fighter.alive and not self.battle.finished:
                    if self.battle.elements.target_state(uid, self.battle.turn)["actions_to_skip"]:
                        masks[team].append(mask)
                        continue
                    for x, y in self.battle.reachable_cells(fighter):
                        if (x, y) != fighter.position:
                            mask[MOVE_START + x * GRID_SIZE + y] = True
                    for slot_index, slot in enumerate(ABILITY_SLOTS):
                        for target_index, target_uid in enumerate(self.global_slots):
                            if target_uid is None:
                                continue
                            action = {"kind": "ability", "slot": slot, "target": target_uid}
                            if self.battle.check_ability(fighter, action)[0] is None:
                                mask[self._ability_code(slot_index, target_index)] = True
                masks[team].append(mask)
        return masks

    def _decode(self, uid, code, mask):
        if type(code) is not int or not 0 <= code < ACTION_SPACE_SIZE or not mask[code]:
            return {"kind": "invalid"}
        if code == DEFEND_ACTION:
            return {"kind": "defend"}
        if MOVE_START <= code < MOVE_END:
            cell = code - MOVE_START
            return {"kind": "move", "destination": [cell // GRID_SIZE, cell % GRID_SIZE]}
        offset = code - ABILITY_START
        slot = ABILITY_SLOTS[offset // GLOBAL_FIGHTER_SLOTS]
        target = self.global_slots[offset % GLOBAL_FIGHTER_SLOTS]
        return {"kind": "ability", "slot": slot, "target": target}

    def decode_joint_actions(self, action_codes):
        if not isinstance(action_codes, dict) or set(action_codes) != {"A", "B"}:
            raise ValueError("Actions must contain lists for both teams A and B.")
        masks = self.action_masks()
        actions = {}
        for team in ("A", "B"):
            codes = action_codes[team]
            if not isinstance(codes, (list, tuple)) or len(codes) != MAX_TEAM_SIZE:
                raise ValueError("Each team must provide exactly four action codes.")
            for index, uid in enumerate(self.team_slots[team]):
                fighter = self.battle.fighters.get(uid) if uid else None
                if fighter is not None and fighter.alive:
                    actions[uid] = self._decode(uid, codes[index], masks[team][index])
        return actions

    def observation_vector(self, perspective):
        """Return a fixed numeric vector with the observing team first."""
        if perspective not in ("A", "B") or self.battle is None:
            raise ValueError("Perspective must be A or B after reset().")
        state = self.battle.observe()
        active_event = state["arena"].get("active_event")
        event_name = active_event.get("name") if active_event else None
        vector = [self.battle.turn / self.max_turns]
        vector.extend(float(state["arena"]["weather"] == name) for name in WEATHERS)
        vector.extend(float(event_name == name) for name in EVENTS)
        vector.extend(float(self.battle.objective.mode == name) for name in OBJECTIVE_TYPES)
        vector.extend(self.battle.objective.scores[t] / self.target_score for t in (perspective, "B" if perspective == "A" else "A"))
        vector.extend(state["synergy"][t]["active_score"] / 100 for t in (perspective, "B" if perspective == "A" else "A"))
        ordered_uids = self.team_slots[perspective] + self.team_slots["B" if perspective == "A" else "A"]
        for uid in ordered_uids:
            fighter = self.battle.fighters.get(uid) if uid else None
            if fighter is None:
                vector.extend([0.0] * 27)
                continue
            elemental = self.elements_for(fighter)
            cooldowns = [min(1.0, max(0, fighter.ready_at[slot] - self.battle.turn) / 5)
                         for slot in ABILITY_SLOTS]
            status = self.battle.elements.target_state(uid, self.battle.turn)
            aura = status["aura"]["element"] if status["aura"] else None
            reduction = status["defense_reduction"]
            vector.extend([
                1.0, float(fighter.alive), fighter.hp / fighter.max_hp,
                fighter.energy / self.battle.rules["maximum_energy"],
                min(1.0, fighter.shield / fighter.max_hp),
                fighter.position[0] / (GRID_SIZE - 1), fighter.position[1] / (GRID_SIZE - 1),
                float(fighter.defending), *cooldowns,
            ])
            vector.extend(float(elemental == element) for element in ELEMENTS)
            vector.extend(float(aura == element) for element in ELEMENTS)
            vector.extend([float(status["actions_to_skip"] > 0),
                           reduction["fraction"] if reduction else 0.0])
        return vector

    @staticmethod
    def elements_for(fighter):
        return fighter.data["element"]

    def observation(self):
        return {
            "state": self.battle.observe(),
            "vectors": {team: self.observation_vector(team) for team in ("A", "B")},
            "action_masks": self.action_masks(),
        }

    def step(self, action_codes):
        if self.battle is None:
            raise ValueError("Call reset() before step().")
        if self.battle.finished:
            raise ValueError("The episode is finished; call reset().")
        actions = self.decode_joint_actions(action_codes)
        record = self.battle.step(actions)
        evaluation = calculate_rewards(self.battle)
        rewards = {team: round(evaluation["teams"][team]["relative_reward"]
                               - self._previous_relative[team], 6) for team in ("A", "B")}
        self._previous_relative = {team: evaluation["teams"][team]["relative_reward"] for team in ("A", "B")}
        turn_limit = self.battle.turn >= self.max_turns and self.battle.reason.startswith("Turn limit reached")
        terminated = self.battle.finished and not turn_limit
        truncated = self.battle.finished and turn_limit
        if not self.battle.finished:
            self.battle.begin_turn()
        info = {"turn_record": record, "evaluation": evaluation,
                "result": self.battle.result() if self.battle.finished else None}
        return self.observation(), rewards, terminated, truncated, info

    def sample_legal_actions(self, rng=None):
        """Sample one legal action code for every team slot."""
        rng = rng or self._rng
        masks = self.action_masks()
        return {team: [rng.choice([code for code, legal in enumerate(mask) if legal])
                       for mask in masks[team]] for team in ("A", "B")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--abilities", type=Path, default=DEFAULT_ABILITIES)
    parser.add_argument("--synergies", type=Path, default=DEFAULT_SYNERGIES)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--team-size", choices=("random", "1", "2", "3", "4"), default="random")
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args()
    try:
        if args.episodes < 1:
            raise ValueError("episodes must be positive.")
        team_size = args.team_size if args.team_size == "random" else int(args.team_size)
        env = GenshinArenaEnv(args.csv, args.abilities, args.synergies, seed=args.seed,
                              team_size=team_size, max_turns=args.max_turns)
        policy_rng = random.Random(f"{args.seed}:random-policy")
        wins = {"A": 0, "B": 0, "Draw": 0}
        returns = []
        for episode in range(1, args.episodes + 1):
            observation, info = env.reset()
            episode_return = 0.0
            done = False
            while not done:
                actions = env.sample_legal_actions(policy_rng)
                observation, rewards, terminated, truncated, step_info = env.step(actions)
                episode_return += rewards["A"]
                done = terminated or truncated
            result = step_info["result"]
            wins[result["winner"] or "Draw"] += 1
            returns.append(episode_return)
            print(f"Episode {episode}: {result['outcome']} in {result['turns']} turns; A return {episode_return:.4f}.")
        print(f"Results: A {wins['A']}, B {wins['B']}, draws {wins['Draw']}.")
        print(f"Average Team A return: {sum(returns) / len(returns):.4f}.")
        print(f"Observation vector length: {len(observation['vectors']['A'])}; action space per slot: {ACTION_SPACE_SIZE}.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
