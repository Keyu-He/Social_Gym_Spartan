# Skull (Skull & Roses)

## Overview
Skull is a pure psychological bluffing game with no luck element. Players take turns placing face-down discs — roses (safe) or a skull (dangerous). Once placing begins, any player can switch to bidding, claiming they can safely flip a certain number of discs across all players' stacks. The highest bidder must then flip that many discs, starting with their own. If they flip a skull, they lose a disc permanently. First to win 1 round wins the game.

## Players
- Number of players: 4 (Alice, Bob, Charlie, Diana)
- Roles: All players are symmetric "Player" roles.

## Objective
Be the first player to win 1 round, or be the last player remaining (others eliminated by losing all their discs).

## How to Play
Each round has three phases:

**Phase 1 — Place:**
- Players take turns in round-robin order, each placing one face-down disc from their hand (`rose` or `skull`).
- After everyone has placed at least 1 disc, any player may start bidding instead of placing by declaring `bid N`.

**Phase 2 — Bid:**
- Players take turns either raising the bid (`bid N`, must be higher than current) or passing (`pass`).
- When all but one player have passed, the remaining bidder wins the bid at their declared number.

**Phase 3 — Flip (winning bidder only):**
- The bidder must flip exactly N discs total.
- **Rule:** They MUST flip all of their own placed discs first before flipping others'.
- To flip: `flip NAME` to reveal the top disc of NAME's stack.
- **Rose revealed:** Continue flipping.
- **Skull revealed:** Round is lost. The bidder permanently loses 1 random disc from their hand.
- **All N discs are roses:** The bidder wins the round (1 win needed to win the game).

After each round, discs are returned to hands and a new round begins. A player with 0 discs is eliminated.

## Scoring / Payoffs
- First player to win 1 round (or last player with discs): +1.0
- Other players: -1.0
- If max turns reached: player with most round wins wins (+1.0)

## Social Skills Tested
- **Psychological bluffing:** Placing a skull to trap a bidder who bids high.
- **Risk assessment:** Deciding how high to bid based on what you placed and what you infer others placed.
- **Deception:** Convincing others you placed roses so they bid high and hit your skull.
- **Opponent modeling:** Tracking other players' bidding and placing patterns to infer skull placement.
