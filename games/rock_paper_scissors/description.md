# Rock Paper Scissors

## Overview
Rock Paper Scissors is the classic hand game where two players simultaneously reveal a shape. Rock beats Scissors, Scissors beats Paper, Paper beats Rock. In this version, players compete over 10 rounds and accumulate points, so every round counts toward the final score.

## Players
- Number of players: 2
- Roles: Both players are symmetric "Player" roles with no information advantage.

## Objective
Accumulate more points than your opponent over 10 rounds. The player with the higher total score wins. If scores are tied, it is a draw.

## How to Play
1. Each round, both players simultaneously choose `rock`, `paper`, or `scissors`.
2. No speaking is allowed — only actions.
3. The round winner earns 1 point; the loser earns 0; a tie earns 0 for both.
4. After 10 rounds, whoever has more total points wins.

## Scoring / Payoffs
Per round:
- Win: +1 point
- Loss: 0 points
- Tie: 0 points

Final rewards: Winner gets +1.0, loser gets -1.0, tie gives 0.0 each.

## Social Skills Tested
- **Pattern recognition:** Detecting and exploiting opponent tendencies over repeated play.
- **Randomization:** Avoiding exploitable patterns in your own choices.
- **Opponent modeling:** Predicting what the opponent will do next based on history.
