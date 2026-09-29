# Genshin Impact Multi-Agent Combat Arena

## Project Objective

This project builds a turn-based, multi-agent combat research environment inspired by Genshin Impact. The objective is to investigate whether an AI can learn when a statistically weaker team should attack, defend, reposition, conserve resources, pursue an objective, or exploit environmental conditions under uncertainty.

The simulator does not treat raw ATK, HP, DEF, rarity, or popularity as sufficient indicators of success. Outcomes are also affected by objectives, weather, dynamic events, elemental reactions, team synergy, positioning, energy, cooldowns, shields, healing, and coordinated actions.

> This is a custom research simulator. Ability values, reaction multipliers, synergy bonuses, arena rules, and environmental modifiers are simplified design assumptions and are not official Genshin Impact mechanics.

## Implemented Scope

- Supports equal-sized 1v1, 2v2, 3v3, and 4v4 battles.
- Loads all 103 character records from the supplied CSV.
- Uses character name, element, weapon, level-90 HP, ATK, DEF, and all listed roles.
- Removes unnamed CSV index columns and converts comma-separated roles into flags.
- Provides custom battle abilities for eight starter characters.
- Uses an 8x8 grid with movement, range, positioning, defence, attacks, skills, bursts, energy, cooldowns, buffs, healing, and shields.
- Implements clear, blizzard, and thunderstorm weather.
- Implements energy-blackout and shield-nullification events.
- Implements Elimination and Control Zone objectives.
- Implements simplified Vaporize, Melt, Overloaded, Electro-Charged, Frozen, and Superconduct reactions.
- Implements configurable pair and role-coverage synergy.
- Provides random, rule-based, Linear Q-learning, and MAPPO-style controllers.
- Supports curriculum training, checkpoint resuming, action masking, and CSV training history.

## System Architecture

```text
Character CSV + Ability/Synergy Rules
                 |
                 v
          Data Preparation
                 |
                 v
       Turn-Based Battle Engine
                 |
       +---------+----------+
       |         |          |
   Weather   Objectives   Events
       |         |          |
       +---- Reactions -----+
                 |
                 v
       Seven-Part Reward Model
                 |
                 v
   RL Environment + Action Masks
                 |
       +---------+----------+
       |         |          |
    Random    Linear Q    MAPPO
                 |
                 v
       Evaluation and CSV Results
```

## Strategic Reward

The agent is evaluated using the competition weights:

| Component | Weight |
|---|---:|
| Survival | 25% |
| Objective control | 20% |
| Resource management | 15% |
| Team synergy | 15% |
| Damage contribution | 10% |
| Adaptability | 10% |
| Risk efficiency | 5% |

Damage contributes only 10%, preventing pure DPS optimisation from dominating the learned strategy.

## Controllers

### Random baseline

Selects uniformly from currently legal actions. It establishes the minimum performance baseline.

### Rule-based baseline

Uses manually written domain knowledge to prioritise objectives, healing, shielding, buffs, attacks, movement, and energy efficiency. It acts as an expert benchmark.

### Linear Q-learning

Estimates action values from a compact set of battle features and learns from the seven-part reward. It is transparent and dependency-free, but it can model only relatively simple relationships.

### MAPPO-style controller

Uses a shared neural actor for all four character slots and a centralized critic for the complete team state. It applies legal-action masking, Generalized Advantage Estimation, clipped PPO updates, and curriculum learning.

The implementation is described as MAPPO-style because Team A is trained against fixed random/rule-based opponents; full two-sided self-play is future work.

## Experimental Results

Each reported evaluation used 100 battles.

| Experiment | A wins | B wins | Draws | A win rate | Average reward |
|---|---:|---:|---:|---:|---:|
| Random baseline comparison | 21 | 23 | 56 | 21% | 0.0009 |
| Rule-based baseline comparison | 88 | 0 | 12 | 88% | 0.3570 |
| Linear Q comparison | 52 | 1 | 47 | 52% | 0.2111 |
| Initial MAPPO vs random | 45 | 9 | 46 | 45% | 0.2037 |
| Initial MAPPO vs rule-based | 9 | 86 | 5 | 9% | -0.2750 |
| Curriculum MAPPO vs random | 65 | 0 | 35 | **65%** | **0.3009** |
| Curriculum MAPPO vs rule-based | 4 | 96 | 0 | 4% | -0.3191 |

Curriculum training improved MAPPO's win rate against random opponents from 45% to 65% and eliminated losses in that evaluation. The rule-based expert remained significantly stronger, showing that learned coordination and expert-opponent generalisation require further work.

Results from separate evaluation programs should be treated as indicative rather than perfectly paired unless they use the same saved scenario list and random seeds.

## Installation

Python 3.10 or newer is recommended.

The battle engine, simulator, Linear Q agent, and comparison tools use the Python standard library. MAPPO additionally requires NumPy and PyTorch:

```powershell
python -m pip install -r requirements_mappo.txt
```

Keep `genshin_impact.csv` in the same directory as the Python files.

## Main Commands

Run a standard battle:

```powershell
python battle_engine.py
```

Run a battle with advanced mechanics:

```powershell
python battle_engine.py --weather blizzard --events --objective control_zone --reactions --synergy --max-turns 100
```

Generate simulation data:

```powershell
python simulation_runner.py --matches 100 --output simulation_results.csv
```

Test the reinforcement-learning environment:

```powershell
python rl_environment.py --episodes 10 --seed 2026
```

Train Linear Q-learning:

```powershell
python train_linear_q.py --episodes 500 --evaluation-episodes 50
```

Compare random, rule-based, and Linear Q controllers:

```powershell
python compare_agents.py --model linear_q_model.json --matches 100
```

Train curriculum MAPPO:

```powershell
python train_mappo.py --curriculum --episodes 1500 --episodes-per-update 10 --model mappo_curriculum.pt --history mappo_training_history.csv
```

Evaluate curriculum MAPPO:

```powershell
python train_mappo.py --evaluate-only --model mappo_curriculum.pt --evaluation-episodes 100
```

Resume interrupted MAPPO training:

```powershell
python train_mappo.py --resume --episodes 500 --model mappo_curriculum.pt --history mappo_training_history.csv
```

## Important Files

| File | Purpose |
|---|---|
| `data_loader.py` | Cleans and prepares the character CSV |
| `abilities.json` | Custom starter-character ability definitions |
| `ability_loader.py` | Validates and combines character and ability data |
| `battle_engine.py` | Resolves movement, abilities, damage, support and outcomes |
| `arena_environment.py` | Weather and dynamic-event logic |
| `arena_objectives.py` | Elimination and Control Zone rules |
| `elemental_system.py` | Elemental auras and reactions |
| `synergies.json` | Configurable synergy rules |
| `synergy_system.py` | Synergy scoring and contextual bonuses |
| `reward_calculator.py` | Seven-part strategic reward calculation |
| `simulation_runner.py` | Reproducible batch simulation |
| `rl_environment.py` | Fixed observation/action interface and action masks |
| `train_linear_q.py` | Linear Q-learning trainer |
| `compare_agents.py` | Fair baseline-comparison experiment |
| `train_mappo.py` | MAPPO-style neural and curriculum trainer |
| `requirements_mappo.txt` | Neural-training dependencies |

## Limitations

- Only eight characters currently have complete simulator abilities.
- Only two objective types and three weather types are implemented.
- Terrain variation and several proposed hazards remain future work.
- The learned agents do not yet consistently defeat the expert rule-based controller.
- Full adversarial self-play, recurrent memory, opponent modelling, and separate actor networks are not implemented.
- Simulator ability values and environmental modifiers require balancing through larger experiments.
- Popularity and banner information remain in the original data but are intentionally excluded from the first combat controllers.

## Future Work

- Add ability definitions for all characters.
- Add survival, escort, relic-protection, and resource-collection objectives.
- Add floating islands, crystalline caverns, dense forest, and ancient ruins.
- Train through two-sided self-play and opponent pools.
- Add recurrent policies for partial observability and event forecasting.
- Tune rewards and hyperparameters across multiple seeds.
- Report confidence intervals and paired statistical comparisons.

## Conclusion

The project demonstrates that context-aware combat can be represented as a multi-agent reinforcement-learning problem rather than a simple stat comparison. Learned controllers improved substantially beyond random behaviour, and curriculum MAPPO reached a 65% win rate against random opponents. The continued advantage of the expert rule-based controller is an important experimental finding: complex objective-aware coordination still requires better training, self-play, and richer policy models.
