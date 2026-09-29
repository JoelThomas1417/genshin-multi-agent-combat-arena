"""Run a Genshin-inspired arena battle with synergy, reactions, and objectives.

Keep this file beside arena_environment.py, arena_objectives.py, elemental_system.py,
synergy_system.py, synergies.json, ability_loader.py, data_loader.py,
abilities.json, and characters_prepared.csv
(or genshin_impact.csv).
No extra packages are needed.

    python battle_engine.py
    python battle_engine.py --team-a noelle gorou --team-b arlecchino raiden_shogun
    python battle_engine.py --seed 42 --max-turns 60 --log match_42.json
    python battle_engine.py --weather blizzard --events
    python battle_engine.py --weather random --events --seed 7
    python battle_engine.py --objective control_zone --weather blizzard --events
    python battle_engine.py --objective control_zone --controller-b combat --max-turns 100
    python battle_engine.py --team-a furina arlecchino --team-b noelle gorou --reactions
    python battle_engine.py --team-a bennett xiangling --team-b noelle gorou --synergy

Starter rules (custom simulator assumptions, not official game mechanics):
* An empty 8x8 grid. Defaults: elimination objective, clear weather, no events.
* Blizzard reduces movement by one cell (minimum one) and boosts Cryo damage
  20%. Thunderstorm boosts Electro damage 30%. Weather is fixed for a match.
* With --events, check every 5 turns: 40% no event, 30% energy blackout, and
  30% shield nullification. An event affects 3 turns, starting immediately.
  Blackout blocks bursts and energy gain, but retains stored energy.
  Nullification removes existing shields and prevents new shield abilities.
* Both sides have the same team size, from 1 to 4. Default: 2v2.
* Every living fighter chooses one action from the same start-of-turn state.
* Resolve movement, then defense/healing/shields/buffs, then attacks.
* Within each phase, interleave teams; A starts odd turns, B starts even turns.
  Seeded shuffling orders teammates. Knocked-out fighters lose pending actions.
* Movement uses orthogonal paths and cannot cross occupied cells. Earlier moves
  win contested destinations. Ability range uses Manhattan distance.
* Raw damage = the caster's effective scaling stat * ability multiplier.
  Damage after defense = raw damage * 1000 / (1000 + effective target DEF).
  Defend reduces that result by 40% for this turn. Shields absorb damage next.
* Healing cannot exceed maximum HP or revive a knocked-out fighter.
* Buffs do not stack: keep one strongest bonus per stat, refreshing equal buffs.
  Weaker buffs are discarded. Shields replace existing shields only if stronger.
* A duration D used on turn T expires at the start of T+D. A cooldown C becomes
  ready at T+C. Invalid actions consume a turn without spending energy.
* With --reactions, elemental hits apply three-turn auras. A compatible second
  element consumes the aura and triggers Vaporize, Melt, Overloaded,
  Electro-Charged, Frozen, or Superconduct. Physical attacks do not affect auras.
  These reaction values are simplified simulator rules. Visibility remains full.
* With --synergy, configured character pairs and balanced role coverage receive
  visible composition scores and combat bonuses. Energy blackout can suppress
  an energy pair; shield nullification can suppress a shield-specific effect.
* Elimination mode: eliminate the opposing team; the turn limit gives a draw.
* Control-zone mode: the central four cells form one zone. After combat, a team
  earns one point if only its living fighters occupy the zone. Empty/contested
  zones award nothing. More occupants do not earn more points. First to 10
  points wins (adjust with --target-score), even with opponents still alive.
  At the turn limit, higher points win; tied points give a draw. Elimination
  also ends this mode and takes precedence over scoring on the final turn.
  Capturing is automatic through occupancy; it requires no separate action.

The included controller reacts to current effects using explicit rules; it is
not trained and does not plan using the event probabilities yet. In control-zone
mode it moves to the zone, prioritizes enemies inside, and repositions within it
to reach enemies. Use --controller-a/--controller-b combat to compare with the
original combat-only baseline. Both are hand-written rules, not learned policies.
Rules and scores are public. Additional objectives
and the seven-part competition score are later additions. battle_log.json records
environment changes, actions, objective scores, outcomes, and state history.

For a future controller, call Battle.begin_turn() to receive new events BEFORE
choosing actions, then Battle.step(actions). Repeated begin_turn() calls return
the same decision state without advancing again. Actions maps fighter IDs such
as 'A:noelle' to dictionaries. Examples:
    {"kind": "ability", "slot": "skill", "target": "A:noelle"}
    {"kind": "move", "destination": [3, 4]}
    {"kind": "defend"}
Omitted fighters defend. Battle.observe() exposes current effects/probabilities,
not the seed, random-generator state, or actual future event outcomes.
"""

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import random

from ability_loader import DEFAULT_ABILITIES, PROJECT_DIR, load_combat_roster
from arena_environment import ArenaEnvironment, WEATHER_RULES
from arena_objectives import BattleObjective, OBJECTIVE_TYPES
from elemental_system import ElementalSystem
from synergy_system import DEFAULT_SYNERGIES, SynergySystem


GRID_SIZE = 8
DEFENSE_SCALE = 1000.0
DEFEND_MULTIPLIER = 0.60


def distance(first, second):
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


@dataclass
class Fighter:
    uid: str
    team: str
    data: dict
    position: tuple
    hp: float
    energy: int
    shield: float = 0.0
    shield_expires: int = 0
    buffs: dict = field(default_factory=dict)
    ready_at: dict = field(default_factory=dict)
    defending: bool = False

    @property
    def alive(self):
        return self.hp > 0

    @property
    def max_hp(self):
        return self.data["lvl_90_HP"]

    def stat(self, name):
        bonus = self.buffs.get(name, {}).get("fraction", 0.0)
        return self.data[name] * (1.0 + bonus)


class Battle:
    """Own match state and resolve actions independently of the controller."""

    def __init__(self, rules, roster, team_a, team_b, seed=42, max_turns=30,
                 weather="clear", events_enabled=False, objective="elimination", target_score=10,
                 controller_a="objective", controller_b="objective", reactions_enabled=False,
                 synergy_enabled=False, synergy_path=DEFAULT_SYNERGIES):
        if len(team_a) != len(team_b) or not 1 <= len(team_a) <= 4:
            raise ValueError("Choose equally sized teams with 1 to 4 characters each.")
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive whole number.")
        if controller_a not in ("objective", "combat") or controller_b not in ("objective", "combat"):
            raise ValueError("Controllers must be objective or combat.")
        for names in (team_a, team_b):
            if len(set(names)) != len(names):
                raise ValueError("A character cannot appear twice on the same team.")
            for name in names:
                if name not in roster:
                    raise ValueError(f"Unknown character {name!r}. Available: {', '.join(roster)}")

        self.rules = deepcopy(rules)
        self.seed = seed
        self.rng = random.Random(seed)
        self.arena = ArenaEnvironment(weather=weather, events_enabled=events_enabled, seed=seed)
        self.objective = BattleObjective(mode=objective, grid_size=GRID_SIZE, target_score=target_score)
        self.elements = ElementalSystem(enabled=reactions_enabled)
        self.synergy = SynergySystem(roster, {"A": team_a, "B": team_b},
                                     enabled=synergy_enabled, rules_path=synergy_path)
        self.baseline_controllers = {"A": controller_a, "B": controller_b}
        self.max_turns = max_turns
        self.turn = 0
        self.finished = False
        self.winner = None
        self.reason = "Match in progress."
        self.fighters = {}
        self.history = []
        self.events = []
        self.controllers_used = set()
        self._turn_started = False
        self._decision_state = None
        self._initiative_order = []
        for team, names, x in (("A", team_a, 1), ("B", team_b, GRID_SIZE - 2)):
            first_y = (GRID_SIZE - len(names)) // 2
            for index, name in enumerate(names):
                data = deepcopy(roster[name])
                uid = f"{team}:{name}"
                initial_shield = float(rules.get("starting_shield", 0))
                self.fighters[uid] = Fighter(
                    uid, team, data, (x, first_y + index), float(data["lvl_90_HP"]),
                    rules["starting_energy"], shield=initial_shield,
                    shield_expires=max_turns + 1 if initial_shield else 0,
                    ready_at={slot: 1 for slot in data["abilities"]},
                )
        self._last_synergy_state = {
            team: self.synergy.state(team, self.arena, self.elements.enabled) for team in ("A", "B")
        }
        self.initial_state = self.observe()

    def observe(self):
        """Return a detached JSON-friendly view; changes cannot mutate the match."""
        return {
            "turn": self.turn,
            "arena": {"width": GRID_SIZE, "height": GRID_SIZE, "objective": self.objective.mode, **self.arena.public_state()},
            "objective": self.objective.public_state(list(self.fighters.values())),
            "synergy": {team: self.synergy.state(team, self.arena, self.elements.enabled)
                        for team in ("A", "B")},
            "awaiting_actions": self._turn_started,
            "fighters": {
                uid: {
                    "name": fighter.data["name"], "team": fighter.team,
                    "element": fighter.data["element"], "weapon": fighter.data["weapon"],
                    "position": list(fighter.position), "alive": fighter.alive,
                    "hp": round(fighter.hp, 4), "max_hp": fighter.max_hp,
                    "energy": fighter.energy, "shield": round(fighter.shield, 4),
                    "shield_expires": fighter.shield_expires, "defending": fighter.defending,
                    "buffs": deepcopy(fighter.buffs), "ready_at": dict(fighter.ready_at),
                    "elemental_status": self.elements.target_state(uid, self.turn),
                    "base_atk": fighter.data["lvl_90_ATK"], "base_def": fighter.data["lvl_90_DEF"],
                    "effective_atk": self.effective_stat(fighter, "lvl_90_ATK"),
                    "effective_def": self.effective_stat(fighter, "lvl_90_DEF") * self.elements.defense_multiplier(uid, self.turn),
                    "movement_range": self.arena.movement_allowance(fighter.data["movement_range"]),
                }
                for uid, fighter in self.fighters.items()
            },
        }

    def log(self, kind, actor, message, **details):
        self.events.append({"turn": self.turn, "kind": kind, "actor": actor, "message": message, **details})

    def living(self, team=None):
        return [fighter for fighter in self.fighters.values() if fighter.alive and (team is None or fighter.team == team)]

    def effective_stat(self, fighter, stat):
        return fighter.stat(stat) * self.synergy.stat_multiplier(
            fighter.team, stat, self.arena, self.elements.enabled)

    def reachable_cells(self, fighter):
        occupied = {other.position for other in self.living() if other.uid != fighter.uid}
        visited = {fighter.position: 0}
        pending = deque([fighter.position])
        while pending:
            x, y = pending.popleft()
            steps = visited[(x, y)]
            if steps >= self.arena.movement_allowance(fighter.data["movement_range"]):
                continue
            for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= cell[0] < GRID_SIZE and 0 <= cell[1] < GRID_SIZE and cell not in occupied and cell not in visited:
                    visited[cell] = steps + 1
                    pending.append(cell)
        return visited

    def check_ability(self, actor, action):
        slot = action.get("slot")
        if not isinstance(slot, str) or slot not in actor.data["abilities"]:
            return "unknown ability", None, None
        ability = actor.data["abilities"][slot]
        environment_error = self.arena.ability_block_reason(slot, ability)
        if environment_error:
            return environment_error, None, None
        target_id = action.get("target")
        target = self.fighters.get(target_id) if isinstance(target_id, str) else None
        if target is None or not target.alive:
            return "target is absent or knocked out", None, None
        if self.turn < actor.ready_at[slot]:
            return "ability is on cooldown", None, None
        if actor.energy < ability["energy_cost"]:
            return "not enough energy", None, None
        allowed = ability["target"]
        if (allowed == "enemy" and target.team == actor.team) or (allowed == "ally" and target.team != actor.team) or (allowed == "self" and target.uid != actor.uid):
            return "invalid target team", None, None
        if distance(actor.position, target.position) > ability["range_cells"]:
            return "target is out of range", None, None
        return None, ability, target

    def predicted_damage(self, actor, target, ability):
        raw = self.effective_stat(actor, ability["scaling_stat"]) * ability["multiplier"]
        raw *= self.arena.damage_multiplier(ability["element"])
        raw *= self.synergy.element_damage_multiplier(actor.team, ability["element"], self.arena, self.elements.enabled)
        effective_def = self.effective_stat(target, "lvl_90_DEF") * self.elements.defense_multiplier(target.uid, self.turn)
        mitigated = raw * DEFENSE_SCALE / (DEFENSE_SCALE + effective_def)
        reaction = self.elements.preview(target.uid, ability["element"], self.turn)
        mitigated *= reaction["damage_multiplier"]
        if reaction["reaction"]:
            mitigated *= self.synergy.effect_multiplier(actor.team, "reaction", self.arena, self.elements.enabled)
        return mitigated * (DEFEND_MULTIPLIER if target.defending else 1.0)

    def execute_ability(self, actor, action):
        error, ability, target = self.check_ability(actor, action)
        if error:
            self.log("invalid", actor.uid, f"{actor.uid} waits: {error}.")
            return
        energy_before = actor.energy
        synergy_gain = ability["energy_gain"] * self.synergy.effect_multiplier(
            actor.team, "energy", self.arena, self.elements.enabled)
        gain = self.arena.energy_gain(int(round(synergy_gain)))
        actor.energy = min(self.rules["maximum_energy"], actor.energy - ability["energy_cost"] + gain)
        actor.ready_at[action["slot"]] = self.turn + ability["cooldown_turns"]
        effect = ability["effect_type"]
        details = {"target": target.uid, "slot": action["slot"], "ability": ability["name"],
                   "energy_spent": ability["energy_cost"],
                   "energy_gained": actor.energy - (energy_before - ability["energy_cost"])}
        prefix = f"{actor.uid} uses {ability['name']} on {target.uid}"
        if effect == "damage":
            effective_def = self.effective_stat(target, "lvl_90_DEF") * self.elements.defense_multiplier(target.uid, self.turn)
            reaction = self.elements.apply(target.uid, ability["element"], self.turn)
            raw = self.effective_stat(actor, ability["scaling_stat"]) * ability["multiplier"]
            raw *= self.arena.damage_multiplier(ability["element"])
            raw *= self.synergy.element_damage_multiplier(actor.team, ability["element"], self.arena, self.elements.enabled)
            damage = raw * DEFENSE_SCALE / (DEFENSE_SCALE + effective_def)
            damage *= reaction["damage_multiplier"]
            if reaction["reaction"]:
                damage *= self.synergy.effect_multiplier(actor.team, "reaction", self.arena, self.elements.enabled)
            damage *= DEFEND_MULTIPLIER if target.defending else 1.0
            absorbed = min(target.shield, damage)
            target.shield = max(0.0, target.shield - absorbed)
            if target.shield == 0:
                target.shield_expires = 0
            hp_loss = min(target.hp, max(0.0, damage - absorbed))
            target.hp = max(0.0, target.hp - hp_loss)
            details.update(element=ability["element"], hp_damage=hp_loss, shield_absorbed=absorbed)
            if self.elements.enabled:
                details.update(reaction=reaction["reaction"],
                               reaction_multiplier=reaction["damage_multiplier"],
                               aura_before=reaction["aura_before"], aura_after=reaction["aura_after"])
            details["weather_multiplier"] = self.arena.damage_multiplier(ability["element"])
            self.log(effect, actor.uid, f"{prefix}: {hp_loss:.1f} HP damage, {absorbed:.1f} absorbed by shield.", **details)
            if reaction["reaction"]:
                self.log("reaction", actor.uid,
                         f"{reaction['reaction']} triggers on {target.uid}. {reaction['description']}",
                         target=target.uid, reaction=reaction["reaction"],
                         incoming_element=ability["element"], aura_element=reaction["aura_before"],
                         damage_multiplier=reaction["damage_multiplier"])
            if not target.alive:
                target.buffs.clear()
                self.elements.clear_target(target.uid)
                self.log("knockout", actor.uid, f"{target.uid} is knocked out.", target=target.uid)
        elif effect == "heal":
            amount = self.effective_stat(actor, ability["scaling_stat"]) * ability["multiplier"]
            amount *= self.synergy.effect_multiplier(actor.team, "heal", self.arena, self.elements.enabled)
            restored = min(amount, target.max_hp - target.hp)
            target.hp += restored
            self.log(effect, actor.uid, f"{prefix}: restores {restored:.1f} HP.", restored=restored, **details)
        elif effect == "shield":
            amount = self.effective_stat(actor, ability["scaling_stat"]) * ability["multiplier"]
            amount *= self.synergy.effect_multiplier(actor.team, "shield", self.arena, self.elements.enabled)
            replaced = amount > target.shield
            if replaced:
                target.shield = amount
                target.shield_expires = self.turn + ability["duration_turns"]
            detail = f"shield is now {target.shield:.1f}" if replaced else "existing shield is at least as strong"
            self.log(effect, actor.uid, f"{prefix}: {detail}.", replaced=replaced, **details)
        elif effect == "buff":
            stat = ability["buff_stat"]
            previous = target.buffs.get(stat, {}).get("fraction", 0.0)
            applied = ability["bonus_fraction"] >= previous
            if applied:
                target.buffs[stat] = {"fraction": ability["bonus_fraction"], "expires": self.turn + ability["duration_turns"]}
            detail = f"{stat} +{target.buffs[stat]['fraction']:.0%}" if applied else "stronger existing buff retained"
            self.log(effect, actor.uid, f"{prefix}: {detail}.", applied=applied, **details)

    def zone_move(self, actor, reserved):
        """Approach the zone while reserving different destinations for allies."""
        if self.objective.mode != "control_zone":
            return None
        reachable = self.reachable_cells(actor)
        candidates = [cell for cell in reachable if ("zone_move", *cell) not in reserved]
        if not candidates:
            return None
        if self.objective.contains(actor.position):
            foes = self.living("B" if actor.team == "A" else "A")
            candidates = [cell for cell in candidates if self.objective.contains(cell)]
            if not foes or not candidates:
                return None
            nearest = lambda cell: min(distance(cell, foe.position) for foe in foes)
            destination = min(candidates, key=lambda cell: (nearest(cell), reachable[cell], cell))
            if nearest(destination) < nearest(actor.position):
                reserved.add(("zone_move", *destination))
                return {"kind": "move", "destination": list(destination)}
            return None
        destination = min(candidates, key=lambda cell: (self.objective.distance_to_zone(cell), reachable[cell], cell))
        if self.objective.distance_to_zone(destination) < self.objective.distance_to_zone(actor.position):
            reserved.add(("zone_move", *destination))
            return {"kind": "move", "destination": list(destination)}
        return None

    def choose_action(self, actor, reserved):
        """A small rule-based baseline; priorities are inspectable and replaceable."""
        foes = [other for other in self.living() if other.team != actor.team]
        objective_aware = self.objective.mode == "control_zone" and self.baseline_controllers[actor.team] == "objective"
        if objective_aware and not any(
                self.objective.contains(ally.position) for ally in self.living(actor.team)):
            move = self.zone_move(actor, reserved)
            if move is not None:
                return move
        candidates = []
        for slot, ability in actor.data["abilities"].items():
            for target in self.living():
                action = {"kind": "ability", "slot": slot, "target": target.uid}
                if self.check_ability(actor, action)[0] is not None:
                    continue
                effect = ability["effect_type"]
                reservation = (effect, target.uid, ability.get("buff_stat", ""))
                if effect != "damage" and reservation in reserved:
                    continue
                near_enemy = any(distance(target.position, foe.position) <= 4 for foe in foes)
                score = 0.0
                if effect == "heal" and target.hp / target.max_hp < 0.65:
                    amount = self.effective_stat(actor, ability["scaling_stat"]) * ability["multiplier"]
                    amount *= self.synergy.effect_multiplier(actor.team, "heal", self.arena, self.elements.enabled)
                    score = 1.25 * min(amount, target.max_hp - target.hp)
                elif effect == "shield" and near_enemy:
                    amount = self.effective_stat(actor, ability["scaling_stat"]) * ability["multiplier"]
                    amount *= self.synergy.effect_multiplier(actor.team, "shield", self.arena, self.elements.enabled)
                    score = 0.55 * max(0.0, amount - target.shield)
                elif effect == "buff" and near_enemy:
                    current = target.buffs.get(ability["buff_stat"], {}).get("fraction", 0.0)
                    if ability["bonus_fraction"] > current:
                        # ATK buffs enhance later hits; DEF buffs help absorb threats.
                        value = self.effective_stat(target, ability["buff_stat"]) * (ability["bonus_fraction"] - current)
                        score = value * min(ability["duration_turns"], 3) * (1.5 if ability["buff_stat"] == "lvl_90_ATK" else 1.0)
                elif effect == "damage":
                    damage = self.predicted_damage(actor, target, ability)
                    score = min(damage, target.hp + target.shield)
                    score *= 1.0 + 0.20 * (1.0 - target.hp / target.max_hp)
                    if objective_aware and self.objective.contains(target.position):
                        score *= 1.5  # Removing a contestant can unlock objective points.
                # Spending energy needs a worthwhile immediate benefit in this baseline.
                if score > 0:
                    score -= ability["energy_cost"] * 2.0
                    candidates.append((score, action, reservation))

        if candidates:
            score, action, reservation = max(candidates, key=lambda item: item[0])
            if score > 0:
                if reservation[0] != "damage":
                    reserved.add(reservation)
                return action

        if objective_aware:
            return self.zone_move(actor, reserved) or {"kind": "defend"}

        # Move toward the nearest opponent when no useful ability is in range.
        if foes:
            foe = min(foes, key=lambda enemy: (distance(actor.position, enemy.position), enemy.hp, enemy.uid))
            reachable = self.reachable_cells(actor)
            destination = min(reachable, key=lambda cell: (distance(cell, foe.position), reachable[cell], cell))
            if distance(destination, foe.position) < distance(actor.position, foe.position):
                return {"kind": "move", "destination": list(destination)}
        return {"kind": "defend"}

    def initiative(self):
        groups = {team: [fighter.uid for fighter in self.living(team)] for team in ("A", "B")}
        for fighters in groups.values():
            self.rng.shuffle(fighters)
        teams = ("A", "B") if self.turn % 2 else ("B", "A")
        return [groups[team][index] for index in range(max(map(len, groups.values()))) for team in teams if index < len(groups[team])]

    def expire_statuses(self):
        for fighter in self.living():
            fighter.defending = False
            if fighter.shield and self.turn >= fighter.shield_expires:
                fighter.shield = 0.0
                fighter.shield_expires = 0
                self.log("expiry", fighter.uid, f"{fighter.uid}'s shield expires.")
            for stat, buff in list(fighter.buffs.items()):
                if self.turn >= buff["expires"]:
                    del fighter.buffs[stat]
                    self.log("expiry", fighter.uid, f"{fighter.uid}'s {stat} buff expires.")
        for expiry in self.elements.advance(self.turn):
            self.log("element_expiry", "arena", expiry["message"],
                     target=expiry["target"], effect=expiry["effect"], name=expiry["name"])

    def begin_turn(self):
        """Apply new events and return the state agents should use for decisions."""
        if self.finished:
            raise ValueError("The match has already ended.")
        if self._turn_started:
            return deepcopy(self._decision_state)
        self.turn += 1
        self.events = []
        self.expire_statuses()
        for announcement in self.arena.advance(self.turn):
            self.log("environment", "arena", announcement["message"],
                     event=announcement["event"], transition=announcement["transition"])
        for team in ("A", "B"):
            current = self.synergy.state(team, self.arena, self.elements.enabled)
            previous = self._last_synergy_state[team]
            if self.synergy.enabled and (current["active_score"] != previous["active_score"]
                                         or current["effects"] != previous["effects"]):
                self.log("synergy_change", "arena",
                         f"Team {team} synergy changes: active score {current['active_score']}.",
                         team=team, previous_score=previous["active_score"],
                         active_score=current["active_score"], effects=deepcopy(current["effects"]),
                         suppressed_pairs=deepcopy(current["suppressed_pairs"]))
            self._last_synergy_state[team] = current
        if self.arena.active("shield_nullification"):
            for fighter in self.living():
                if fighter.shield > 0:
                    removed = fighter.shield
                    fighter.shield = 0.0
                    fighter.shield_expires = 0
                    self.log("shield_removed", "arena", f"Shield nullification removes {fighter.uid}'s shield ({removed:.1f} points).",
                             target=fighter.uid, removed=removed)
        self._turn_started = True
        self._decision_state = self.observe()
        self._initiative_order = self.initiative()
        return deepcopy(self._decision_state)

    def step(self, actions=None):
        """Resolve one turn; the baseline automatically begins its decision phase.

        External agents should call begin_turn() before selecting their actions.
        Supplying actions without doing so remains supported, but those actions
        will have been chosen without seeing that turn's newly sampled event.
        """
        if self.finished:
            raise ValueError("The match has already ended.")
        if actions is not None:
            if not isinstance(actions, dict) or any(uid not in self.fighters for uid in actions):
                raise ValueError("actions must map known fighter IDs to action dictionaries.")
            actions = deepcopy(actions)
        self.controllers_used.add("supplied_joint_actions" if actions is not None else "rule_based_baseline")
        before = self.begin_turn()
        order = self._initiative_order
        if actions is None:
            reserved = {"A": set(), "B": set()}
            actions = {uid: self.choose_action(self.fighters[uid], reserved[self.fighters[uid].team]) for uid in order}
        else:
            actions = {uid: actions.get(uid, {"kind": "defend"}) for uid in order}

        phases = {"move": [], "support": [], "damage": []}
        for uid in order:
            action = actions[uid]
            if not isinstance(action, dict):
                self.log("invalid", uid, f"{uid} waits: action must be a dictionary.")
                continue
            kind = action.get("kind")
            if kind == "move":
                phases["move"].append(uid)
            elif kind == "defend":
                phases["support"].append(uid)
            elif kind == "ability":
                slot = action.get("slot")
                ability = self.fighters[uid].data["abilities"].get(slot) if isinstance(slot, str) else None
                if ability:
                    phases["damage" if ability["effect_type"] == "damage" else "support"].append(uid)
                else:
                    self.log("invalid", uid, f"{uid} waits: unknown ability.")
            else:
                self.log("invalid", uid, f"{uid} waits: unknown action kind.")

        for phase in ("move", "support", "damage"):
            for uid in phases[phase]:
                actor = self.fighters[uid]
                if not actor.alive:
                    self.log("cancelled", uid, f"{uid}'s action is cancelled after knockout.")
                    continue
                if self.elements.consume_skipped_action(uid):
                    self.log("frozen", uid, f"{uid} loses this action because of Frozen.")
                    continue
                action = actions[uid]
                if action["kind"] == "move":
                    destination = action.get("destination")
                    valid = isinstance(destination, (list, tuple)) and len(destination) == 2 and all(type(value) is int for value in destination)
                    destination = tuple(destination) if valid else None
                    if destination is None or destination == actor.position or destination not in self.reachable_cells(actor):
                        self.log("invalid", uid, f"{uid} cannot move: destination is blocked, unchanged, or unreachable.")
                    else:
                        previous = actor.position
                        actor.position = destination
                        self.log("move", uid, f"{uid} moves from {previous} to {destination}.", origin=list(previous), destination=list(destination))
                elif action["kind"] == "defend":
                    actor.defending = True
                    self.log("defend", uid, f"{uid} defends: incoming damage is reduced by 40% this turn.")
                else:
                    self.execute_ability(actor, action)

        self.finished, self.winner, self.reason, objective_events = self.objective.resolve_turn(
            self.turn, list(self.fighters.values()), at_turn_limit=self.turn >= self.max_turns)
        for event in objective_events:
            self.log("objective", "arena", event["message"],
                     **{key: value for key, value in event.items() if key != "message"})
        self._turn_started = False
        record = {"turn": self.turn, "before": before, "initiative": list(order), "actions": actions, "events": deepcopy(self.events), "after": self.observe()}
        self.history.append(record)
        return deepcopy(record)

    def result(self):
        synergy_scores = {team: self.synergy.state(team, self.arena, self.elements.enabled)["active_score"]
                          for team in ("A", "B")}
        return {
            "finished": self.finished, "winner": self.winner,
            "outcome": (f"Team {self.winner} wins" if self.winner else "Draw") if self.finished else "In progress",
            "reason": self.reason, "turns": self.turn,
            "objective": self.objective.mode,
            "scores": dict(self.objective.scores),
            "synergy_scores": synergy_scores,
            "survivors": {team: [fighter.data["name"] for fighter in self.living(team)] for team in ("A", "B")},
        }

    def export_log(self):
        return deepcopy({
            "log_version": 5, "seed": self.seed, "max_turns": self.max_turns,
            "controllers_used": sorted(self.controllers_used), "rules": self.rules,
            "baseline_controllers": dict(self.baseline_controllers),
            "engine_rules": {"grid_size": GRID_SIZE, "defense_scale": DEFENSE_SCALE, "defend_multiplier": DEFEND_MULTIPLIER,
                             "phases": ["movement", "support", "damage", "objective"],
                             "turn_limit_result": self.objective.configuration()["turn_limit_result"],
                             "elemental_reactions": self.elements.enabled,
                             "team_synergy": self.synergy.enabled},
            "arena_configuration": self.arena.configuration(),
            "objective_configuration": self.objective.configuration(),
            "elemental_configuration": self.elements.configuration(),
            "synergy_configuration": self.synergy.configuration(),
            "templates": {uid: fighter.data for uid, fighter in self.fighters.items()},
            "initial_state": self.initial_state, "turns": self.history, "result": self.result(),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="Prepared or original character CSV")
    parser.add_argument("--abilities", type=Path, default=DEFAULT_ABILITIES)
    parser.add_argument("--team-a", nargs="+", default=["noelle", "xiangling"])
    parser.add_argument("--team-b", nargs="+", default=["arlecchino", "gorou"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--objective", choices=OBJECTIVE_TYPES, default="elimination")
    parser.add_argument("--target-score", type=int, default=10, help="Points needed to win control-zone mode")
    parser.add_argument("--controller-a", choices=("objective", "combat"), default="objective",
                        help="Use zone-aware rules or the original combat-only rules for Team A")
    parser.add_argument("--controller-b", choices=("objective", "combat"), default="objective",
                        help="Use zone-aware rules or the original combat-only rules for Team B")
    parser.add_argument("--weather", choices=[*WEATHER_RULES, "random"], default="clear",
                        help="Fixed weather, or randomly select it at match initialization")
    parser.add_argument("--events", action="store_true", help="Enable unpredictable three-turn arena events")
    parser.add_argument("--reactions", action="store_true", help="Enable simplified elemental auras and reactions")
    parser.add_argument("--synergy", action="store_true", help="Enable configured pair and role-coverage bonuses")
    parser.add_argument("--synergies", type=Path, default=DEFAULT_SYNERGIES, help="Synergy rules JSON file")
    parser.add_argument("--log", type=Path, default=Path("battle_log.json"))
    parser.add_argument("--quiet", action="store_true", help="Show only the final result")
    args = parser.parse_args()
    try:
        protected = [args.abilities, PROJECT_DIR / "characters_prepared.csv", PROJECT_DIR / "genshin_impact.csv",
                     PROJECT_DIR / "data_loader.py", PROJECT_DIR / "ability_loader.py",
                     PROJECT_DIR / "arena_environment.py", PROJECT_DIR / "arena_objectives.py",
                     PROJECT_DIR / "elemental_system.py", PROJECT_DIR / "synergy_system.py",
                     PROJECT_DIR / "synergies.json", Path(__file__)]
        protected.append(args.synergies)
        if args.csv is not None:
            protected.append(args.csv)
        for path in protected:
            if args.log.resolve() == path.resolve() or (args.log.exists() and path.exists() and args.log.samefile(path)):
                raise ValueError("Choose a log filename that does not overwrite a source or project file.")
        rules, roster = load_combat_roster(args.csv, args.abilities)
        battle = Battle(rules, roster, args.team_a, args.team_b, args.seed, args.max_turns,
                        weather=args.weather, events_enabled=args.events,
                        objective=args.objective, target_score=args.target_score,
                        controller_a=args.controller_a, controller_b=args.controller_b,
                        reactions_enabled=args.reactions, synergy_enabled=args.synergy,
                        synergy_path=args.synergies)
        if not args.quiet:
            print(f"Starter arena: {battle.arena.weather}, 8x8 grid, {battle.objective.mode} objective.")
            print(WEATHER_RULES[battle.arena.weather]["description"])
            print("Random events enabled: checks every 5 turns; events last 3 turns." if args.events else "Random events disabled.")
            print("Elemental reactions enabled: auras last 3 turns." if args.reactions else "Elemental reactions disabled.")
            if args.synergy:
                for team in ("A", "B"):
                    state = battle.synergy.state(team, battle.arena, battle.elements.enabled)
                    print(f"Team {team} synergy: {state['active_score']} points; effects {state['effects']}.")
            else:
                print("Team synergy disabled.")
            print(f"Rule-based controllers: A={args.controller_a}, B={args.controller_b}. Ability values are custom simulator assumptions.")
            print(f"Team A: {', '.join(args.team_a)}")
            print(f"Team B: {', '.join(args.team_b)}")
            if battle.objective.mode == "control_zone":
                print(f"Control zone: {list(battle.objective.zone_cells)}. First to {args.target_score} points wins.")
        while not battle.finished:
            record = battle.step()
            if not args.quiet:
                print(f"\nTurn {battle.turn}")
                for event in record["events"]:
                    print("  " + event["message"])
                for fighter in battle.fighters.values():
                    print(f"  {fighter.uid}: HP {fighter.hp:.0f}/{fighter.max_hp}, energy {fighter.energy}, shield {fighter.shield:.0f}")
                if battle.objective.mode == "control_zone":
                    print(f"  Zone score: A {battle.objective.scores['A']} | B {battle.objective.scores['B']}")
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as output:
            json.dump(battle.export_log(), output, indent=2)
        result = battle.result()
        print(f"\n{result['outcome']} after {battle.turn} turns. {result['reason']}")
        if battle.objective.mode == "control_zone":
            print(f"Final zone score: A {battle.objective.scores['A']} | B {battle.objective.scores['B']}")
        print(f"Battle log saved to: {args.log.resolve()}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
