# Minority Game (El Farol Bar Problem)

## Overview
The Minority Game, inspired by the El Farol Bar problem, is a multi-player anti-coordination game. Players must independently decide to "go" or "stay," and whichever choice fewer than half the players make is the winning side. No communication is allowed — success requires predicting what the majority will do and doing the opposite.

## Players
- Number of players: 5
- Roles: All players are symmetric "Player" roles with no special information.

## Objective
Maximize your own cumulative score over 12 rounds. The top half of players (by score) win, the bottom half lose.

## How to Play
1. Each round, all 5 players simultaneously choose `go` or `stay`.
2. No speaking is allowed — only actions.
3. The minority group (fewer than half of all players, i.e., 0, 1, or 2 out of 5 choosing the same option) wins 3 points each.
4. The majority group scores 0 for that round.
5. After 12 rounds, players are ranked by cumulative score; top half wins, bottom half loses.

## Scoring / Payoffs
Per round:
- Players in the minority group: +3 points each
- Players in the majority group: 0 points

Win conditions:
- Top half of players by score: +1.0
- Bottom half: -1.0
- Middle (if odd number of players): 0.0
- Players tied on score share one result: a tied group inside the top half wins, inside the bottom half loses, and one straddling a boundary draws (0.0). Seating order never decides. The description, the goal and every turn's instruction state this rule, with two worked examples (30, 30, 30, 12, 12: three draws and two losses; 30, 20, 20, 12, 12: a win, two draws and two losses).
- All tied: Draw (0.0 each)

The game description, the goal line and each turn's instruction all state this one rule.

## Unclear replies
A reply that does not name exactly one of `go` or `stay` (including a failed model call) is not read as a move. A move word right after a negation in its clause ("don't go", "I won't stay") does not name that move, so "I won't stay, I go" is go. The engine says so to everyone and asks that player once more; every other move stays hidden, because replies in this state are private and the round results report only the parsed moves. If the second reply is also unclear, the move is drawn at random, each move equally likely, from a stream seeded by the roster. The round results mark a drawn move. The match still lasts 12 rounds; a round with a repeated reply takes one extra step.

## Social Skills Tested
- **Anti-coordination:** Unlike most games where you want to match the group, here you must predict and avoid the majority choice.
- **Crowd psychology modeling:** Estimating what most agents will do in order to deliberately do the opposite.
- **Decentralized equilibrium:** Whether a stable distribution emerges (roughly half go / half stay) without communication.
- **Adaptive strategy:** Adjusting behavior when your previous choice became the majority.
