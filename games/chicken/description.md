# Chicken (Hawk-Dove)

## Overview
Chicken (also called Hawk-Dove) is a game of brinkmanship. Two players each choose whether to "swerve" (back down) or go "straight" (hold firm). If both hold firm, they crash and both lose. If one swerves, the one who goes straight wins big. The game tests nerve, bluffing, and credible commitment.

## Players
- Number of players: 2
- Roles: Both players are symmetric "Player" roles.

## Objective
Maximize your own score over 10 rounds. Higher total score wins.

## How to Play
1. Each round, both players simultaneously choose `swerve` or `straight`.
2. No speaking is allowed — only actions.
3. Points are awarded based on the combination of choices (see payoffs below).
4. After 10 rounds, whoever has more total points wins.

## Scoring / Payoffs
Per round payoffs (Player 1, Player 2):
- Both swerve: 3 each
- One swerves, one goes straight: 1 for swerver, 5 for straight
- Both straight: 0 each (crash!)

Win conditions:
- Higher scorer wins (+1.0), lower loses (-1.0)
- Tied scores: Draw (0.0 each); the rules text says so

## Unclear replies
A reply that does not name exactly one of `swerve` or `straight` (including a failed model call) is not read as a move. A move word right after a negation in its clause (within the three words before it: "I will not swerve", "don't swerve") does not name that move, so "I will not swerve, I go straight" is straight and "I will not swerve" alone is unclear; the rules text says so. The engine says so to both players and asks that player once more; the other player's move stays hidden, because replies in this state are private and the round results report only the parsed moves. If the second reply is also unclear, the move is drawn at random, each move equally likely, from a stream seeded by the roster. The round results mark a drawn move. The match still lasts 10 rounds; a round with a repeated reply takes one extra step.

## Social Skills Tested
- **Brinkmanship:** Committing credibly to going straight to force the opponent to swerve.
- **Risk calibration:** Knowing when to back down to avoid a costly mutual crash.
- **Exploitation vs. safety:** Balancing the high reward of going straight against the risk of collision.
