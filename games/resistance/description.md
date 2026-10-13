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
2. **Mission Proposal:** The current leader proposes a team using `propose NAME1 NAME2 ...`. Team sizes: [2, 3, 2, 3, 3] for missions 1-5. Names are read as whole words in the order given (commas, "and" and full stops do not matter), from "propose" to the end of that sentence. A proposal of the wrong size, a missing proposal (a failed model call) or one that names nobody is announced, and the same leader proposes once more. If the second proposal is also the wrong size, the engine trims it to the first players named or fills it up with players in seat order who are not on it, announces the team, and the team goes to the vote; this is not a rejection. The official rules have no such default, and the rules text says so. Every proposal replaces the previous team.
3. **Mission Vote:** All players vote `approve` or `reject`. The first approve/reject word in the reply decides, and a negated one counts the other way ("I do not approve" rejects); "yes" or "no" count only when neither word is there. A reply that is neither, or a failed model call, counts as a reject and the game says so, so a team needs a majority of all five players (a tie rejects).
   - Majority approves: proceed to Mission Execute.
   - Majority rejects: leader rotates to next player, return to Discussion. 5 consecutive rejections = Spies win.
4. **Mission Execute:** Each team member secretly plays `succeed` or `fail`. Number of fail cards is announced publicly (not who played them). Each member is told privately how its card was read: the first card word decides, a negated one counting the other way ("I will not fail" is succeed); Resistance members always play succeed, and a Spy's reply that is neither card, or a failed call, counts as succeed. The mission resolves after its one turn, so cards never carry over to another team.
   - Any fail card = mission fails.
   - All succeed cards = mission succeeds.

Five consecutive rejected proposals in one round hand the game to the Spies, as in the official rules (five teams rejected in a single round win the game for the Spies); the engine counts rejections in a row and resets the count when a team is approved. The longest possible game is 205 turns: five missions, each with four rejected proposals and an approved one, every proposal asked twice (5 Discussion turns, 2 proposal turns and 1 vote) and one execution turn per mission; it is 180 turns when no proposal is asked again. The engine's 205-turn cap is a backstop that never cuts a game, so the models are not told about it. (It was 100 turns, which cut any game with 14 or more proposals.)

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
