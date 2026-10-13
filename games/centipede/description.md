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
2. In each game, the current active player chooses `take` or `pass`. The game's first mover moves at nodes 1, 3 and 5, the second mover at nodes 2, 4 and 6.
3. `take`: The game ends immediately. Both players receive the payoff for the current node.
4. `pass`: The pot structure changes to the next node (with different payoffs), and the other player gets their turn.
5. The game also ends if all 6 nodes are passed through (players receive the pass-through payoff).
6. Cumulative payoffs across all 4 games determine the winner.

The whole payoff table below is in the game description the models read (complete information), and each
turn's instruction repeats the remaining nodes in the mover's own terms. A reply that names neither `take`
nor `pass`, or both (or a failed model call), is not read as a move; a move word right after a negation
("I will not take") does not name that move. The engine says so publicly, without quoting the reply, and the
same player chooses again. If the second reply is also unclear, the move is drawn at random, take and pass
equally likely, from a stream seeded by the roster.

The players cannot talk: the Turn state is private, so the other player sees only the parsed move ("Alice
passes!", "Alice TAKES at node 3!"), never the reply's text. Each player keeps its own replies in its history.
Equal cumulative scores are a draw, and the rules text says so.

## Scoring / Payoffs
Node payoffs (first mover, second mover) of that game if "take" is chosen at that node. Whoever takes
gets the larger share of the node, so backward induction predicts a take at node 1 (the textbook
centipede of Rosenthal 1981 and McKelvey and Palfrey 1992):
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
