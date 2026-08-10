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
- All tied: Draw (0.0 each)

## Social Skills Tested
- **Trust:** Relying on all other players to cooperate, knowing one defection ruins it.
- **Coordination under uncertainty:** Aligning on the risky-but-rewarding equilibrium vs. the safe equilibrium.
- **Collective action:** Maintaining group cooperation over repeated rounds without communication.
