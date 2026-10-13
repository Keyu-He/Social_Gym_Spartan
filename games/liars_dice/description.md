# Liar's Dice

## Overview
Liar's Dice is a bluffing and probability game where each player holds hidden dice. Players take turns making increasingly bold claims about how many dice showing a particular face value exist across ALL players' dice combined. On your turn you can challenge the current claim by calling "liar," forcing an immediate reveal.

## Players
- Number of players: 3 (Alice, Bob, Charlie)
- Roles: All players are symmetric "Player" roles. No teams.

## Objective
Be the last player with at least one die remaining. Eliminate opponents by catching their bluffs or by making bids they can't beat.

## How to Play
1. At the start of each round, each player privately rolls all their remaining dice (only they can see their results). The rolls come from the roster seed, so a roster replays the same dice for the same moves.
2. Players take turns in round-robin order.
3. On your turn, you must either:
   - **Bid:** Claim that at least X dice showing face value Y exist across all players' dice. Your bid must be higher than the previous one (higher quantity, or same quantity with a higher face value). Use `bid QUANTITY FACE` (e.g., `bid 3 4` means "at least three 4s").
   - **Call Liar:** Challenge the current bid using `liar`. All dice are revealed immediately.
   - Only the move the reply commits to counts (`committed_move.py`, shared by the bluffing games): "Liar! I don't believe that bid." and "I call liar on your bid" are liar calls. A move the reply mentions while negating it, making it hypothetical or conditional, asking about it, putting it on a later turn, or talking about it as a subject does not count: "I'm not going to call liar; bid 4 5", "Calling liar is too risky here, so bid 4 5" and "bid 3 4, or should I call liar?" are bids, and "I won't call liar, raise to 3 4" is no move (an illegal move). "a liar", "liars" and "Liar's Dice" are not calls. "dudo", "challenge" and a bare "call" ("I call", "I call your bid") are liar calls; "call for", "call out", "call off" and "call it a" are not. "Bid: 3 4", "bid 3x4", "bid 3 4s", "bid 3 of 4s" and "bid three fours" are read as `bid 3 4`.
   - Anything else is an **illegal move and costs one die**: a bid that is not higher than the current bid, a malformed command, calling `liar` with no bid on the table, a reply that commits to two different moves (a bid and `liar`, or two different bids), or making no move at all. Like a lost challenge, the penalty ends the round at once (no dice are revealed): everyone re-rolls and the offender opens the next round, or the next player in seat order if the offender was eliminated. So a die is only ever lost at the end of a round, and nobody is asked to raise or challenge their own bid. Without this rule a player who never makes a legal bid can never be challenged and never loses dice (this is how the first 10-model run was decided in 58% of games).
4. **Resolution when "liar" is called:**
   - Count actual total dice showing the bid face across all alive players.
   - If actual count < bid quantity: the bidder was lying → bidder loses 1 die.
   - If actual count ≥ bid quantity: the caller was wrong → caller loses 1 die.
5. A new round begins with all surviving players re-rolling their dice. The player who lost the die opens it; if that player was eliminated, the next living player in seat order (the one to its left) opens.
6. A player with 0 dice is eliminated.

## Scoring / Payoffs
- Winner (last player with dice): complete_rating = 1.0
- Eliminated players: complete_rating = -1.0 split among them (-0.5 each with 3 players)

## Social Skills Tested
- **Bluffing:** Making believable bids even when your dice are unfavorable.
- **Probabilistic reasoning:** Estimating the likelihood a bid is valid given your own dice.
- **Risk calibration:** Deciding when the odds make challenging worthwhile vs. continuing to bid.
- **Reading opponents:** Inferring whether an opponent's bid is a stretch based on betting patterns.
