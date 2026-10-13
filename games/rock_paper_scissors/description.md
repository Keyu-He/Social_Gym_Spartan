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

Final rewards: Winner gets +1.0, loser gets -1.0, tie gives 0.0 each; the rules text says equal points are a draw. The engine stores the round points as stated (1 for the round winner, 0 for the loser), and "[Game] Entering state: Round" is posted only when a new round starts.

## Unclear replies
A reply that does not name exactly one throw (including a failed model call) is not read as a move. A throw right after a negation in its clause ("not rock", "I won't play scissors") does not name that throw, so "not rock, paper" is paper. The engine says so to both players and asks that player once more; the other player's throw stays hidden, because replies in this state are private and the round results report only the parsed moves. If the second reply is also unclear, the move is drawn at random, each throw equally likely, from a stream seeded by the roster. The round results mark a drawn move. The match still lasts 10 rounds; a round with a repeated reply takes one extra step.
Singular and plural name the same throw (`rocks`, `papers`, `scissor`).

## Social Skills Tested
- **Pattern recognition:** Detecting and exploiting opponent tendencies over repeated play.
- **Randomization:** Avoiding exploitable patterns in your own choices.
- **Opponent modeling:** Predicting what the opponent will do next based on history.
