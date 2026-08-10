# Liar's Dice

## Overview
Liar's Dice is a bluffing and probability game where each player holds hidden dice. Players take turns making increasingly bold claims about how many dice showing a particular face value exist across ALL players' dice combined. Anyone can challenge a claim at any time by calling "liar," forcing an immediate reveal.

## Players
- Number of players: 3 (Alice, Bob, Charlie)
- Roles: All players are symmetric "Player" roles. No teams.

## Objective
Be the last player with at least one die remaining. Eliminate opponents by catching their bluffs or by making bids they can't beat.

## How to Play
1. At the start of each round, each player privately rolls all their remaining dice (only they can see their results).
2. Players take turns in round-robin order.
3. On your turn, you must either:
   - **Bid:** Claim that at least X dice showing face value Y exist across all players' dice. Your bid must be higher than the previous one (higher quantity, or same quantity with a higher face value). Use `bid QUANTITY FACE` (e.g., `bid 3 4` means "at least three 4s").
   - **Call Liar:** Challenge the current bid using `liar`. All dice are revealed immediately.
4. **Resolution when "liar" is called:**
   - Count actual total dice showing the bid face across all alive players.
   - If actual count < bid quantity: the bidder was lying → bidder loses 1 die.
   - If actual count ≥ bid quantity: the caller was wrong → caller loses 1 die.
5. A new round begins with all surviving players re-rolling their dice.
6. A player with 0 dice is eliminated.

## Scoring / Payoffs
- Winner (last player with dice): complete_rating = 1.0
- Eliminated players: complete_rating = -1.0
- If max turns reached: player with most dice wins (+1.0), others lose (-1.0)

## Social Skills Tested
- **Bluffing:** Making believable bids even when your dice are unfavorable.
- **Probabilistic reasoning:** Estimating the likelihood a bid is valid given your own dice.
- **Risk calibration:** Deciding when the odds make challenging worthwhile vs. continuing to bid.
- **Reading opponents:** Inferring whether an opponent's bid is a stretch based on betting patterns.
