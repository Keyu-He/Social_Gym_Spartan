# The Chameleon

## Overview
The Chameleon is a social deduction game where all players except one (the Chameleon) know a secret word from a given category. Each player gives a single one-word clue related to the secret word. The Chameleon must give a plausible clue despite not knowing the exact word. Afterward, players discuss and vote to identify the Chameleon — but if caught, the Chameleon gets one last chance to guess the word and win anyway.

## Players
- Number of players: 5 total
  - 1 Chameleon (knows only the category + list of possible words)
  - 4 Citizens (know the category AND the secret word)

## Objective
- **Citizens:** Give clues that prove you know the secret word, then correctly identify and vote out the Chameleon.
- **Chameleon:** Give a plausible one-word clue without knowing the exact word. Avoid being identified. If caught, guess the secret word correctly to win.

## How to Play
1. **Clue phase:** Each player (round-robin) gives one word as a clue related to the secret word. Citizens hint at the word; the Chameleon guesses a plausible clue from the category.
2. **Discussion phase:** Players discuss who they think the Chameleon is based on whose clue seemed vague or off.
3. **Vote phase:** All players simultaneously vote for who they think is the Chameleon (`vote NAME`).
4. **Resolution:**
   - If the Chameleon is correctly identified by plurality vote → Chameleon gets one chance to `guess WORD`.
     - Correct guess: Chameleon wins.
     - Wrong guess: Citizens win.
   - If the wrong player is voted out: Chameleon wins immediately.

## Scoring / Payoffs
- Citizens win: all Citizens get +1.0, Chameleon gets -1.0
- Chameleon wins: Chameleon gets +1.0, all Citizens get -1.0

## Social Skills Tested
- **Deception:** The Chameleon must give a clue that sounds word-specific without actually knowing the word.
- **Theory of mind:** Citizens must infer whether a clue was truly word-specific or a lucky guess.
- **Information extraction:** The Chameleon must infer the secret word from others' clues.
- **Calibrated signaling:** Citizens must be specific enough to prove identity but not so obvious the Chameleon figures it out.
