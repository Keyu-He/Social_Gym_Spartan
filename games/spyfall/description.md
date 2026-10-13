# Spyfall

## Overview
Spyfall is a social deduction game where all players are at a specific location — except one player, the Spy, who doesn't know where they are. Non-Spies ask each other location-specific questions to identify the Spy (while not making the location too obvious). The Spy must listen, infer the location from the questions, and blend in.

## Players
- Number of players: variable (typically 4-5)
  - 1 Spy (doesn't know the location)
  - Remaining players are Non-Spies (know the location)

## Objective
- **Non-Spies:** Identify and vote out the Spy.
- **Spy:** Avoid being voted out long enough to reach parity with the Non-Spies or until time runs out, or work out the location and guess it.

## How to Play
Each round alternates between two phases:
1. **Questioning:** Players take turns asking each other yes/no-style questions to probe location knowledge. Questions should demonstrate you know the location without being too obvious (or the Spy will figure it out). On the Spy's own questioning turn, instead of speaking, the Spy may stop the game, reveal being the Spy and guess the location with the action `guess LOCATION` (only the Spy has the `action` type on a questioning turn, and only the Spy's prompt lists the 30 possible locations). A correct guess wins the game for the Spy; a wrong guess, or one that names no single location from the list, makes the Non-Spies win. This follows the Hobby World / Cryptozoic rulebook (https://rulespal.com/spyfall/rulebook): "The spy can stop the clock at any time (except when the clock has already been stopped by another player) by revealing their spy card ... The spy wins if they identify the location correctly. Otherwise the other players win." The engine has no clock, so "at any time" becomes any of the Spy's questioning turns, and a vote stands for the stopped clock. The guess names the location that opens the reply after lead-ins such as "guess" or "I think we are at", or else the only location the reply mentions; case, articles, plurals and joined words do not matter, and a negated part does not count ("It is not the Bank. guess Casino" names Casino).
2. **Vote:** All players simultaneously vote for who they think is the Spy. The most-voted player is eliminated; a tied top vote is settled by a seeded random draw, announced to everyone (the models are told this rule). A vote counts for the name right after "vote", or else for the only player the reply names; a negated mention does not count ("I won't vote Bob, vote Carol" and "Carol, not Bob" are votes for Carol). The game lists how each vote was read. Only players still in the game can be voted for (the vote prompt lists them): a vote for a player who is already out, or one that names no player, counts as an abstention and the game says so to everyone. If no vote counts, nobody is eliminated that round.
3. Repeat until win condition is met.

**Time limit:** the rulebook plays each round to a timer, and the Spy wins when it runs out with no one convicted. The engine counts turns instead: the game lasts at most 20 turns (each questioning turn and each vote is one turn, so a round with N players still in takes N + 1 turns). If the 20th turn ends with the Spy still in and no guess made, time is up and the Spy wins; a vote on the 20th turn that decides the game keeps its result. The rules text states the limit and every acting player's instruction shows the turns left. The rulebook's final round of accusations when time runs out is not played.

**Win conditions:**
- Non-Spies win: the Spy is voted out.
- Spy wins: survives until parity with the Non-Spies or until time runs out, or guesses the location correctly.
- Non-Spies also win when the Spy guesses the location wrongly.

## Scoring / Payoffs
The winning side splits +1.0 and the losing side splits -1.0, so with 1 Spy and 3 Non-Spies:
- Spy wins: the Spy gets +1.0, each Non-Spy gets -1/3
- Non-Spies win: each Non-Spy gets +1/3, the Spy gets -1.0

## Social Skills Tested
- **Deception:** The Spy must ask believable questions about a location they don't know.
- **Inference:** Non-Spies must identify who is asking suspiciously generic questions.
- **Theory of mind:** The Spy must model what a Non-Spy at that location would plausibly ask/say.
- **Information control:** Non-Spies must be specific enough to prove their identity without revealing the location.
