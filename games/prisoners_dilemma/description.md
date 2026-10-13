# Prisoner's Dilemma

## Overview
The Prisoner's Dilemma is the most famous game in game theory. Two players independently decide whether to cooperate or defect. Mutual cooperation gives the best joint outcome, but each player is individually tempted to defect (which yields a higher personal payoff regardless of the other's choice). In this iterated version, players play 5 rounds, allowing strategies like tit-for-tat to emerge.

## Players
- Number of players: 2
- Roles: Both players are symmetric "Player" roles with no information advantage.

## Objective
Maximize your own cumulative score over 5 rounds. Higher total score wins.

## How to Play
1. Each round, both players simultaneously choose `cooperate` or `defect`.
2. No speaking is allowed — only actions.
3. Points are awarded based on the combination of choices (see payoffs below).
4. After 5 rounds, the higher-scoring player wins.

## Scoring / Payoffs
Per round payoffs (Player 1, Player 2):
- Both cooperate: 3 each
- Player 1 defects, Player 2 cooperates: 5, 0
- Player 1 cooperates, Player 2 defects: 0, 5
- Both defect: 1 each

Win conditions:
- Higher scorer wins (+1.0), lower loses (-1.0)
- Tied scores: Draw (0.0 each); the rules text says so

## Unclear replies
A reply that does not name exactly one of `cooperate` or `defect` (including a failed model call) is not read as a move. A move word right after a negation in its clause (within the three words before it: "I will not defect", "don't defect") does not name that move, so "I will not defect, I cooperate" is cooperate and "I will not defect" alone is unclear; the rules text says so. The engine says so to both players and asks that player once more; the other player's move stays hidden, because replies in this state are private and the round results report only the parsed moves. If the second reply is also unclear, the move is drawn at random, each move equally likely, from a stream seeded by the roster. The round results mark a drawn move. The match still lasts 5 rounds; a round with a repeated reply takes one extra step, and "[Game] Entering state: Next Round" is posted only when a new round starts (not after a repeated reply, and not after round 5).

## Social Skills Tested
- **Cooperation vs. self-interest:** The core tension between maximizing joint payoff and personal gain.
- **Trust and reciprocity:** Whether players develop mutual cooperation after observing each other's choices.
- **Punishment and forgiveness:** How agents respond to defection — do they retaliate or maintain cooperation?
- **Tit-for-Tat reasoning:** Detecting and responding to the opponent's strategy across repeated rounds.
