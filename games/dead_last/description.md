# Dead Last

## Overview
Dead Last is a social elimination game about forming alliances and surviving votes. Players discuss openly, then simultaneously point at someone to eliminate. If a strict majority agrees on the same target, that player is eliminated. When three players remain, survivors negotiate and vote on how to split the prize pool — but only if all proposals match exactly.

## Players
- Number of players: 6
- Roles: All players are symmetric "Player" roles with no hidden information.

## Objective
Survive to the final round and negotiate the largest share of the 100-point prize. Eliminated players get nothing.

## How to Play
**Elimination phase (repeat until three players remain):**
1. **Discussion:** Each living player speaks once, in seat order — form alliances, argue for targets, negotiate deals.
2. **Point:** All players simultaneously use `point NAME` to vote for someone to eliminate.
   - A point names the first player of the roster (the pointer and eliminated players included) written as a whole word after the first "point" or "pointing" in the reply; a name written with its capital letter is tried before a lowercase one. A reply without "point" or "pointing", or with no player after it, names nobody. The Point instruction states this rule.
   - If a strict majority (more than half of alive players) agree on one target: that player is eliminated.
   - If no majority: no elimination this round.
   - A point that names the pointer, an eliminated player or nobody does not count; the pointer gets a private notice.

**Final phase (when three players remain; at most one player is eliminated per round, so the final always has three):**
3. **Final Negotiation:** Each survivor speaks once, in seat order, about how to split 100 points; then the binding offers are due.
4. **Final Offer:** All survivors simultaneously use `offer X Y Z` to propose their desired split: one whole number per survivor, in the order the engine lists the survivors (seat order), summing to 100.
   - Shares may instead be labelled with names, in any order and in either direction (`offer Carol 25, Alice 50, Bob 25`, `50 for Alice, 25 for Bob, 25 for Carol`, or a mix): an offer that names every survivor gives each one the number written next to its name. Precisely, such an offer is read only by its labels: each survivor gets the one number written right before or right after its name, in the same sentence, with no other name or number in between, and each number goes to one survivor. If the numbers cannot be paired with the names that way, or can be paired in more than one way that gives different splits, the offer is unclear and counts as a mismatch. An offer that names only some survivors is read in seat order.
   - The result line lists every offer as read (by name or in seat order).
   - If ALL proposals match exactly: points are distributed as proposed.
   - If any mismatch: everyone gets 0. An offer that does not list exactly one whole, non-negative number per survivor summing to 100 (or no offer) counts as a mismatch; the result line names it.
   - Offers are made once: the final is decided in the step all survivors submit.

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
