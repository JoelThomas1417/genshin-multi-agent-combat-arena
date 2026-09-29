"""Victory rules for the starter arena; these are custom simulator rules.

elimination:
    Eliminate every opponent. The turn limit gives a draw.
control_zone:
    The four central cells form one zone. After every turn's combat, a team
    gains one point if it has at least one living fighter inside and the other
    team has none. Extra occupants do not add extra points. Contested and empty
    zones award nothing. Occupying the zone requires no separate action.
    First to target_score wins, even with enemies alive. At the turn limit,
    higher points win; equal points give a draw. Elimination ends either mode
    immediately and takes precedence over scoring on that final turn.

Scores are separate from the proposed seven-part competition evaluation.
All objective rules, occupancy, and scores are visible to both teams.
"""


OBJECTIVE_TYPES = ("elimination", "control_zone")


class BattleObjective:
    """Resolve victory using living fighters with uid, team, and position."""

    def __init__(self, mode="elimination", grid_size=8, target_score=10):
        if mode not in OBJECTIVE_TYPES:
            raise ValueError(f"Unknown objective {mode!r}. Choose elimination or control_zone.")
        if type(grid_size) is not int or grid_size < 4 or grid_size % 2:
            raise ValueError("grid_size must be an even whole number of at least four.")
        if type(target_score) is not int or target_score < 1:
            raise ValueError("target_score must be a positive whole number.")
        self.mode = mode
        self.target_score = target_score
        middle = grid_size // 2
        self.zone_cells = tuple((x, y) for x in (middle - 1, middle)
                                for y in (middle - 1, middle)) if mode == "control_zone" else ()
        self.scores = {"A": 0, "B": 0}
        self._last_resolved_turn = 0

    def contains(self, position):
        return tuple(position) in self.zone_cells

    def distance_to_zone(self, position):
        if not self.zone_cells:
            raise ValueError("The elimination objective has no control zone.")
        return min(abs(position[0] - x) + abs(position[1] - y)
                   for x, y in self.zone_cells)

    def public_state(self, fighters):
        fighters = tuple(fighters)
        occupants = {team: [f.uid for f in fighters
                            if f.alive and f.team == team and self.contains(f.position)]
                     for team in ("A", "B")}
        present = [team for team in ("A", "B") if occupants[team]]
        return {
            "type": self.mode,
            "zone_cells": [list(cell) for cell in self.zone_cells],
            "scores": dict(self.scores),
            "target_score": self.target_score if self.mode == "control_zone" else None,
            "occupants": occupants,
            "controller": present[0] if len(present) == 1 else None,
            "zone_status": ("contested" if len(present) == 2 else "controlled" if present else "empty")
                           if self.mode == "control_zone" else "not_applicable",
        }

    def resolve_turn(self, turn, fighters, at_turn_limit=False):
        """Score once after combat. Return (finished, winner, reason, events)."""
        if type(turn) is not int or turn != self._last_resolved_turn + 1:
            raise ValueError("Resolve the objective once per turn, in order.")
        self._last_resolved_turn = turn
        fighters = [fighter for fighter in fighters if fighter.alive]
        surviving = [team for team in ("A", "B") if any(f.team == team for f in fighters)]
        if len(surviving) < 2:
            winner = surviving[0] if surviving else None
            return True, winner, "Opposing team eliminated." if winner else "Both teams eliminated.", []

        events = []
        if self.mode == "control_zone":
            state = self.public_state(fighters)
            owner = state["controller"]
            if owner is not None:
                self.scores[owner] += 1
                message = f"Team {owner} holds the control zone and earns 1 point."
            elif state["zone_status"] == "contested":
                message = "Control zone is contested; neither team earns a point."
            else:
                message = "Control zone is empty; neither team earns a point."
            events.append({"message": message, "scoring_team": owner,
                           "points_awarded": 1 if owner else 0,
                           "scores": dict(self.scores), "zone_status": state["zone_status"]})
            if owner is not None and self.scores[owner] >= self.target_score:
                return True, owner, f"Team {owner} reached {self.target_score} control-zone points.", events

        if at_turn_limit:
            if self.mode == "control_zone":
                if self.scores["A"] != self.scores["B"]:
                    winner = max(self.scores, key=self.scores.get)
                    return True, winner, "Turn limit reached; higher control-zone score wins.", events
                return True, None, "Turn limit reached with equal control-zone scores.", events
            return True, None, "Turn limit reached before elimination.", events
        return False, None, "Match in progress.", events

    def configuration(self):
        return {
            "type": self.mode,
            "zone_cells": [list(cell) for cell in self.zone_cells],
            "target_score": self.target_score if self.mode == "control_zone" else None,
            "points_per_uncontested_turn": 1 if self.mode == "control_zone" else 0,
            "scoring_phase": "after_combat",
            "elimination_ends_match": True,
            "turn_limit_result": "higher_objective_score_or_draw" if self.mode == "control_zone" else "draw",
        }
