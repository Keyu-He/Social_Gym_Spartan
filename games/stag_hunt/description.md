# Stag Hunt

## Overview
Stag Hunt is a cooperation and trust game. Players must decide whether to hunt a stag together (high reward, requires full cooperation) or hunt a hare alone (safe but lower reward). If even one player breaks rank and hunts a hare, the stag hunters get nothing. This game tests trust and collective coordination.

## Players
- Number of players: 4
- Roles: All players are symmetric "Player" roles.

## Objective
Maximize your own score over 10 rounds. Top half of players by score win; bottom half lose.

## How to Play
1. Each round, all 4 players simultaneously choose `stag` or `hare`.
2. No speaking is allowed — only actions.
3. If ALL 4 players choose stag: each gets 5 points.
4. If anyone chooses hare: stag hunters get 0, hare hunters each get 2.
5. After 10 rounds, players are ranked by cumulative score; top half wins, bottom half loses.

## Scoring / Payoffs
Per round:
- All choose stag: 5 points each
- Any hare chosen: hare hunters get 2, stag hunters get 0

Win conditions:
- Top half of players by score: +1.0
- Bottom half: -1.0
- Players tied on score share one result: a tied group inside the top half wins, inside the bottom half loses, and one straddling the boundary draws (0.0). Seating order never decides. The description, the goal and every turn's instruction state this rule, with two worked examples (30, 30, 30, 12: three draws and a loss; 30, 20, 20, 12: a win, two draws and a loss).
- All tied: Draw (0.0 each)

## Unclear replies
A reply that does not name exactly one of `stag` or `hare` (including a failed model call) is not read as a move. A move word right after a negation in its clause ("not hare", "I won't hunt stag") does not name that move, so "not hare, stag" is stag. The engine says so to everyone and asks that player once more; every other move stays hidden, because replies in this state are private and the round results report only the parsed moves. If the second reply is also unclear, the move is drawn at random, each move equally likely, from a stream seeded by the roster. The round results mark a drawn move. The match still lasts 10 rounds; a round with a repeated reply takes one extra step.
Because replies are private, no player's free text reaches the other hunters; they see only the round's choices.

## Social Skills Tested
- **Trust:** Relying on all other players to cooperate, knowing one defection ruins it.
- **Coordination under uncertainty:** Aligning on the risky-but-rewarding equilibrium vs. the safe equilibrium.
- **Collective action:** Maintaining group cooperation over repeated rounds without communication.
