# Survivor

## Overview
Survivor is a social strategy game inspired by the TV show. Players form alliances, vote out opponents one by one, and compete to reach the final round. The twist: eliminated players become jurors who choose the final winner. To win, you need allies to survive the votes — but you also need to avoid making enemies who will refuse to give you the jury victory.

## Players
- Number of players: variable (configurable)
- Roles: All players are symmetric "Player" roles with no hidden information.

## Objective
Survive to the final round (last 3 standing) and win the jury vote. You must balance making allies (to avoid being voted out) against managing your reputation with those you vote out (who will judge you at the end).

## How to Play
**Elimination phase (repeat until three players remain):**
1. **Discussion:** All living players speak once each, in seat order — form alliances, make arguments, lobby for targets.
2. **Vote:** All players simultaneously vote to eliminate one person with `vote NAME`. Most votes = eliminated; a tied top vote is settled by a seeded random draw among the tied players, announced to everyone (the show instead re-votes; if still tied, the tied players are safe and the other players draw rocks). Eliminated players join the jury.
   - A ballot is read among all the game's players, including the voter and eliminated players: the first player named, as a whole word, after the word "vote" or "voting" in the same sentence (a sentence ends at ".", "!" or "?"), trying each "vote" in turn; failing that, the first player named anywhere after the first "vote". A ballot with the word "vote" but no name after it names nobody; a ballot without the word is read as the first player it names. At each step a name written with its capital letter is tried before a lowercase one. Punctuation, quotes and markdown around the name are fine. The Vote and Jury_vote instructions state this rule word for word.
   - A ballot that names the voter, an eliminated player, or nobody does not count; the voter gets a private notice. If no ballot counts, nobody is eliminated and everyone is told.

**Final phase:**
3. **Jury Plea:** The three finalists each make one plea to the jury, in seat order.
4. **Jury Vote:** Each juror votes for one finalist with `vote NAME`, read as above (a ballot that names a non-finalist, or nobody, does not count, with a private notice). Most jury votes wins; finalists tied for the most votes share the win equally. If no juror's ballot counts, all finalists are tied.

## Scoring / Payoffs
- Jury-vote winner: +1.0 (finalists tied for the most jury votes share it: 0.5 each for two, 1/3 each for three)
- Other finalists: 0.0
- Eliminated players (jurors): -1.0

## Social Skills Tested
- **Alliance formation:** Building coalitions to protect yourself while targeting others.
- **Long-term reputation management:** Balancing strategic betrayals against maintaining goodwill with future jurors.
- **Persuasion:** Making compelling arguments in jury pleas about why you played the best game.
- **Social awareness:** Reading who is aligned with whom and who is a threat vs. a potential ally.
