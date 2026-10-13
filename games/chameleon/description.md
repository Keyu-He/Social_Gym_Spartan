# The Chameleon

## Overview
The Chameleon is a social deduction game where all players except one (the Chameleon) know a secret word from a given category. Each player gives a single one-word clue related to the secret word. The Chameleon must give a plausible clue despite not knowing the exact word. Afterward, players discuss and vote to identify the Chameleon — but if caught, the Chameleon gets one last chance to guess the word and win anyway.

## Players
- Number of players: 5 total
  - Everyone sees the same category and 16-word topic card.
  - 1 Chameleon does not know which word on the card is the secret.
  - 4 Citizens know the secret word.
  - The first player in roster order is the dealer and gives the first clue.

## Objective
- **Citizens:** Give clues that prove you know the secret word, then correctly identify and vote out the Chameleon.
- **Chameleon:** Give a plausible one-word clue without knowing the exact word. Avoid being identified. If caught, guess the secret word correctly to win.

## How to Play
1. **Clue phase:** Starting with the dealer, each player in roster order gives one word as a clue related to the secret word. Repeating another player's clue is allowed. Citizens hint at the word; the Chameleon gives a plausible clue from the topic card.
2. **Discussion phase:** Players discuss who they think the Chameleon is based on whose clue seemed vague or off.
3. **Vote phase:** All players simultaneously vote for who they think is the Chameleon (`vote NAME`). The player with the most votes is accused. If the highest count is tied, only the dealer acts in a **TieBreak** phase and chooses one of the tied players with `vote NAME`.
4. **Resolution:**
   - If the Chameleon is correctly identified by plurality vote → Chameleon gets one chance to `guess WORD`.
     - Correct guess: Chameleon wins.
     - Wrong guess: Citizens win.
   - If the wrong player is voted out: Chameleon wins immediately.

## Scoring / Payoffs
The winning side splits `+1.0` and the losing side splits `-1.0`, so with the default
1 Chameleon / 4 Citizens roster:
- Citizens win: each Citizen gets +0.25, Chameleon gets -1.0
- Chameleon wins: Chameleon gets +1.0, each Citizen gets -0.25

Only the sign matters downstream: `experiments/calculate_elo.py` ranks a pair by
comparing rewards, not by their magnitude.

## Original rules and benchmark adaptations

The public topic card, repeated clues, dealer's deciding vote, and caught
Chameleon's single escape guess follow the
[Big Potato rule sheet](https://www.themindcafe.sg/wp-content/uploads/2019/03/The-Chameleon.pdf).
The rule sheet also permits custom topic cards. These are our existing custom
categories, each reduced from 20 to its first 16 entries; they are not a copy of
the published deck.

Each benchmark episode is one five-player round. Seeded sampling replaces dice
and code cards; shuffled roster order replaces seating. Discussion is limited
to one turn per player. The optional multi-round point system and dealer rotation
are not used. A caught Chameleon should name the word as printed on the card;
guess matching is case-insensitive, without a semantic judge or synonym aliases.

How replies are read (every case is shown in the transcript). A clue is the
whole reply without a leading command word ("clue sleet", "Clue: drizzle" and
"My clue is fog" give "sleet", "drizzle", "fog"); a clue of more than one word
stands as given and the game says it is more than one word; a missing clue is
announced. A vote counts for the name right after "vote" ("vote Bob.",
"vote for 'Bob'", "vote **Bob**"), or else for the only player the reply
names; a negated mention does not count ("I won't vote Bob, vote Carol" and
"Carol, not Bob" are votes for Carol); anything else counts as no vote, and the vote results list every
voter with the player the vote was read as or the reason it counts as none.
The dealer's tie-break is read the same way. A caught Chameleon's guess is the
first clause of the reply that says something once lead-ins such as "guess",
"I think the word is", "It's" or "My guess:", quotes and an article are
removed; it must be a card word or a plural, hyphen or space variant of exactly
one card word, and otherwise a reply that mentions exactly one card word names
that word. A negated part does not count, so "It's not hail, it's sleet" is
the guess sleet. A reply
that names no single card word is a wrong guess.

Secret-word clues and self-votes remain accepted model actions. The engine does
not reject a revealing clue. An absent or invalid dealer choice falls back to a
seeded random choice among the tied players, with a public notice and
`tie_break_fallback` flag in internal state. This is a benchmark fallback for
malformed actions, not a tabletop rule; the rules text and the dealer's
tie-break instruction tell the models about it. An absent guess counts as incorrect.
If no valid votes are submitted, all players are tied at zero and the dealer
chooses among them before resolution.

Generate fresh rosters with `python -m experiments.generate_sim_rosters` for this
rule version and record the game configuration and code revision. Some old
rosters contain only agents and a seed; they do not restore the original rules
when run against a newer checkout.

## Social Skills Tested
- **Deception:** The Chameleon must give a clue that sounds word-specific without actually knowing the word.
- **Theory of mind:** Citizens must infer whether a clue was truly word-specific or a lucky guess.
- **Information extraction:** The Chameleon must infer the secret word from others' clues.
- **Calibrated signaling:** Citizens must be specific enough to prove identity but not so obvious the Chameleon figures it out.
