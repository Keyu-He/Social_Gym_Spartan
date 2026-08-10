# The Resistance

## Overview
The Resistance is a social deduction game with no player elimination. A group of Resistance members must complete 3 secret missions, but 2 hidden Spies are trying to sabotage them. A leader proposes a team for each mission; everyone votes to approve or reject. Once a team is approved, members secretly choose to succeed or fail the mission. Resistance members must always succeed; Spies may choose to fail. The number of fail votes is revealed publicly (but not who played them).

## Players
- Number of players: 5 total
  - 3 Resistance members (do not know who the Spies are)
  - 2 Spies (know each other's identities)

## Objective
- **Resistance:** Get 3 missions to succeed.
- **Spies:** Get 3 missions to fail, or cause 5 consecutive proposal rejections.

## How to Play
The game cycles through mission rounds until 3 missions succeed or 3 missions fail:

1. **Discussion:** All players speak publicly to discuss strategy, share suspicions, and argue about who to include/exclude from the team.
2. **Mission Proposal:** The current leader proposes a team using `propose NAME1 NAME2 ...`. Team sizes: [2, 3, 2, 3, 3] for missions 1-5.
3. **Mission Vote:** All players vote `approve` or `reject`.
   - Majority approves: proceed to Mission Execute.
   - Majority rejects: leader rotates to next player, return to Discussion. 5 consecutive rejections = Spies win.
4. **Mission Execute:** Each team member secretly plays `succeed` or `fail`. Number of fail cards is announced publicly (not who played them).
   - Any fail card = mission fails.
   - All succeed cards = mission succeeds.

## Scoring / Payoffs
Team total reward is +1.0 (winning team) / -1.0 (losing team), split equally among team members.
- 3 missions succeed: Resistance wins (+1/3 each), Spies lose (-1/2 each)
- 3 missions fail: Spies win (+1/2 each), Resistance loses (-1/3 each)
- 5 consecutive proposal rejections: Spies win (+1/2 each), Resistance loses (-1/3 each)

## Social Skills Tested
- **Deception:** Spies must vote approve on good teams and argue convincingly to avoid suspicion.
- **Trust and exclusion:** Resistance must identify and exclude Spies from teams.
- **Social inference:** Reading vote patterns and mission outcomes to deduce Spy identities.
- **Coalition building:** Forming trusted sub-groups to reliably pass missions.
