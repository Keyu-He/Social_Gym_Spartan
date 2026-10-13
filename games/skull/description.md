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
- After everyone has placed at least 1 disc, any player may start bidding instead of placing by declaring `bid N`. An opening bid above the discs on the table opens at that number (and goes straight to the flip), said publicly.
- Only the move a reply commits to counts (`committed_move.py`, shared by the bluffing games): a disc or bid the reply negates, makes hypothetical or conditional, asks about, or puts on a later turn does not count ("I don't want to place my skull yet, so rose" is a rose; "I'll place a rose now and bid 2 next turn" is a rose). A disc named with a count ("I have 3 roses and 1 skull") describes the hand.
- Asking for a disc the player no longer holds places the other kind, with a private notice.
- A reply the engine cannot use (no disc and no bid, two different moves, a bid before every player has placed, a disc with an empty hand, or no reply at all) is announced publicly and the player is asked once more. If the second reply cannot be used either, a disc drawn at random from the player's hand is placed (from its own stream of the roster seed, `f"{seed}-place"`); the public line does not say which disc, the player is told privately. With an empty hand the bidding opens at 1 instead. The rules text states all of this.

**Phase 2 — Bid:**
- Bidding goes clockwise, starting with the player after the one who opened it. Each player either raises the bid (`bid N`, must be higher than current) or passes (`pass`).
- A pass is final for the round; players who passed take no further turns in it. Anything other than a higher bid or `pass` counts as a pass, including a reply that commits to both ("bid 4 or pass"); "I could bid 5, but I'll pass" is a pass. A bid above the discs on the table is lowered to that number, said publicly, as the opening bid is. The current high bidder takes no turn while it is highest.
- When all but one player have passed, or as soon as a bid equals the number of discs on the table, that bidder wins the bid at their declared number.

**Phase 3 — Flip (winning bidder only):**
- The bidder must flip exactly N discs total.
- **Rule:** They MUST flip all of their own placed discs first before flipping others'.
- To flip: `flip NAME` to reveal the top disc of NAME's stack. The stack is the player named right after a committed `flip` ("Alice flips Charlie's disc" is Charlie, "I'm not going to flip Bob; flip Diana" is Diana), else the first player the reply names; a negated name ("not Bob") does not count. `flip mine` / `my own` names the bidder's stack. While the bidder still has discs of its own, any named stack flips the bidder's own (the only legal flip).
- **Rose revealed:** Continue flipping.
- **Skull revealed:** Round is lost. The bidder permanently loses 1 random disc from their hand. An invalid flip (no stack with discs left, or two different stacks named) gets a notice; a second one in the same challenge also loses the round.
- **All N discs are roses:** The bidder wins the round (1 win needed to win the game).

After each round, discs are returned to hands and a new round begins; the bidder of the last round places first (if that player was eliminated by another player's skull, the owner of that skull does). A player with 0 discs is eliminated.

## Scoring / Payoffs
- First player to win 1 round (or last player with discs): +1.0
- Other players: -1/3 each (the loss is split, so the rewards sum to 0)

## Social Skills Tested
- **Psychological bluffing:** Placing a skull to trap a bidder who bids high.
- **Risk assessment:** Deciding how high to bid based on what you placed and what you infer others placed.
- **Deception:** Convincing others you placed roses so they bid high and hit your skull.
- **Opponent modeling:** Tracking other players' bidding and placing patterns to infer skull placement.
