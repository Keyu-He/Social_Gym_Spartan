# Undercover

## Overview
Undercover is a social deduction game where most players share one secret word, but two players (Undercovers) have a different but related word. As in the official rules, nobody is told which side they are on: each player sees only their own word and must work out from the others' descriptions whether it is the majority word. Players take turns describing their word in one sentence, trying to show they hold the majority word without giving it away. After descriptions, players vote to eliminate a player they suspect holds the other word.

## Players
- Number of players: 6 total
  - 4 Civilians (majority word)
  - 2 Undercovers (minority word)

## Objective
- **Civilians:** Eliminate both Undercovers.
- **Undercovers:** Survive until exactly 1 Undercover and 1 Civilian remain (or no Civilian remains), while inferring the Civilians' word.

Every player gets the same rules text and the same goal line ("Work out from the descriptions whether your word is the majority (Civilian) word or the minority (Undercover) word ..."); the only difference between a Civilian's prompt and an Undercover's is the word itself. When a player is voted out, everyone is told whether they were a Civilian or an Undercover, as the common rules play it, and the rules text says so. (Until 2026-10-04 each player's goal line named its side, which the official rules do not do.)

## How to Play
Each round alternates between two phases:
1. **Description:** Players take turns giving one sentence describing their secret word — vague enough to not reveal it to the Undercovers, but specific enough to prove they know it.
2. **Vote:** All players simultaneously vote for a player to eliminate. The player with the most votes is eliminated and their side is announced; a tied top vote is settled by a seeded random draw, announced to everyone (the models are told this rule). A vote counts for the name right after "vote" (quotes, markdown or a trailing "'s" do not matter), or else for the only player the reply names; a negated mention does not count ("I won't vote Bob, vote Carol" is a vote for Carol). The game lists how each vote was read. Only players still in the game can be voted for (the vote prompt lists them): a vote for a player who is already out, or one that names no player, counts as an abstention and the game says so to everyone. If no vote counts, nobody is eliminated that round.
3. Repeat until win condition is met.

**Win conditions:**
- Civilians win: all Undercovers are eliminated.
- Undercovers win: the game reaches exactly 1 Undercover and 1 Civilian remaining, or no Civilian remains.

## Scoring / Payoffs
The winning side splits +1.0 and the losing side splits -1.0, so with 2 Undercovers and 4 Civilians:
- Civilians win: each Civilian +1/4, each Undercover -1/2
- Undercovers win: each Undercover +1/2, each Civilian -1/4

A 100-turn cap ends an undecided game as a draw (every player 0). It is a backstop against a game in which nobody is ever voted out, so the models are not told about it.

## Social Skills Tested
- **Deception:** Undercovers must describe a different word while sounding like they know the majority word.
- **Inference:** Civilians must detect which descriptions are slightly "off."
- **Theory of mind:** Both sides must model what the other team knows and how they'll interpret descriptions.
- **Strategic vagueness:** Being specific enough to prove group membership but vague enough to not help the enemy.
