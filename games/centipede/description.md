# Centipede Game

## Overview
The Centipede Game is a sequential bargaining game with a "growing pot." Two players alternate deciding whether to `take` (end the game now and claim the current pot according to preset payoffs) or `pass` (let the pot grow and give the turn to the opponent). Taking early is safe but low-reward; passing is risky but grows the prize. The tension: rational backward induction says take immediately, but greed and trust can lead to better outcomes if both players keep passing.

## Players
- Number of players: 2
- Roles: Both players are symmetric "Player" roles, alternating turns.

## Objective
Maximize your cumulative score across 4 games. First mover alternates each game, so each player starts twice. The player with the higher total wins.

## How to Play
1. Players play 4 independent games of Centipede. First mover alternates each game.
2. In each game, the current active player chooses `take` or `pass`.
3. `take`: The game ends immediately. Both players receive the payoff for the current node.
4. `pass`: The pot structure changes to the next node (with different payoffs), and the other player gets their turn.
5. The game also ends if all 6 nodes are passed through (players receive the pass-through payoff).
6. Cumulative payoffs across all 4 games determine the winner.

## Scoring / Payoffs
Node payoffs (Player 1, Player 2) if "take" is chosen at that node:
- Node 1: (2, 0)
- Node 2: (1, 4)
- Node 3: (6, 2)
- Node 4: (5, 10)
- Node 5: (14, 8)
- Node 6: (13, 24)
- Pass-through (both pass all nodes): (24, 16)

Win conditions:
- Higher cumulative score wins (+1.0), lower loses (-1.0), tied = draw (0.0)

## Social Skills Tested
- **Trust:** Deciding whether to risk passing and trusting the opponent to eventually stop.
- **Backward induction:** Reasoning about what the opponent will do at future nodes.
- **Greed vs. cooperation:** Choosing between taking a safe payoff now vs. waiting for a larger pot.
