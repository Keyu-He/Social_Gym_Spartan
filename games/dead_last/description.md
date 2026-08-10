# Dead Last

## Overview
Dead Last is a social elimination game about forming alliances and surviving votes. Players discuss openly, then simultaneously point at someone to eliminate. If a strict majority agrees on the same target, that player is eliminated. When only 2-3 players remain, survivors negotiate and vote on how to split the prize pool — but only if all proposals match exactly.

## Players
- Number of players: 6
- Roles: All players are symmetric "Player" roles with no hidden information.

## Objective
Survive to the final round and negotiate the largest share of the 100-point prize. Eliminated players get nothing.

## How to Play
**Elimination phase (repeat until 2-3 players remain):**
1. **Discussion:** All living players speak freely — form alliances, argue for targets, negotiate deals.
2. **Point:** All players simultaneously use `point NAME` to vote for someone to eliminate.
   - If a strict majority (more than half of alive players) agree on one target: that player is eliminated.
   - If no majority: no elimination this round.

**Final phase (when 2-3 players remain):**
3. **Final Negotiation:** Survivors discuss how to split 100 points.
4. **Final Offer:** All survivors simultaneously use `offer X Y` (or `offer X Y Z` for 3 players) to propose their desired split. Shares must sum to 100.
   - If ALL proposals match exactly: points are distributed as proposed.
   - If any mismatch: everyone gets 0.

## Scoring / Payoffs
- Eliminated players: -1.0 reward
- Final survivors:
  - If split agreed: each gets their agreed share / 100 (e.g., 34 points → +0.34)
  - If split fails: 0.0 (no reward)

## Social Skills Tested
- **Coalition formation:** Building alliances to direct majority votes.
- **Negotiation:** Reaching a mutually agreeable split under coordination constraints.
- **Strategic targeting:** Deciding who to eliminate to gain negotiating leverage in the final.
- **Credible commitment:** Making deals during discussion that influence voting behavior.
