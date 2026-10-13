# Battle of the Sexes

## Overview
Battle of the Sexes is a classic coordination game where two players prefer different outcomes but both prefer coordinating over not coordinating at all. One player prefers "opera" and the other prefers "football," but meeting at the same event is better than going to different events alone.

## Players
- Number of players: 2
- Roles: Both players are "Player" roles, but they have opposite preferences: the first-listed player (seat 0) prefers opera, the second (seat 1) prefers football. The rules text the models read names the two seated players.

## Objective
Maximize your own score over 10 rounds. Higher total score wins.

## How to Play
1. Each round has two phases:
   - **Discuss**: Players take turns speaking to negotiate which venue to attend.
   - **Choose**: Players simultaneously and privately choose `opera` or `football`.
2. Points are awarded based on the combination of choices (see payoffs below).
3. After 10 rounds, the player with the higher total score wins.

## Scoring / Payoffs
Per round payoffs (seat 0, seat 1):
- Both opera: seat 0 gets 3, seat 1 gets 2
- Both football: seat 0 gets 2, seat 1 gets 3
- One opera, one football: 0 each (miscoordination)

Win conditions:
- Higher scorer wins (+1.0), lower loses (-1.0)
- Tied scores: Draw (0.0 each); the rules text says so

## Unclear replies
A Choose reply that does not name exactly one of `opera` or `football` (including a failed model call) is not read as a choice; a reply naming both venues ("I see your point about football, but let's go with opera") is unclear too. A venue right after a negation in its clause ("not football", "I won't go to opera") does not name that venue, so "not football, opera" is opera. The engine says so to both players and asks that player once more; the other player's choice stays hidden, because Choose replies are private and the round results report only the parsed venues. If the second reply is also unclear, the venue is drawn at random, each venue equally likely, from a stream seeded by the roster, so neither seat's venue is a default. The round results mark a drawn venue. The match still lasts 10 rounds; a round with a repeated choice takes one extra step.

A Discuss message that fails (a failed model call, or a reply that is not a message) is not asked again: both players see "<name>: did nothing" for that turn and the round moves on. The rules text says so.

## Social Skills Tested
- **Coordination:** Aligning on the same event despite differing preferences.
- **Negotiation:** Using the discussion phase to convince the other player.
- **Compromise:** Learning to alternate between preferred outcomes across rounds.
