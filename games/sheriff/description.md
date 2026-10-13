# Sheriff of Nottingham

## Overview
Sheriff of Nottingham is a bluffing and negotiation game set in a medieval market. Players act as merchants who must get goods past a rotating Sheriff. Merchants secretly pack either honest goods (safe but less profitable) or contraband (risky but highly profitable). The Sheriff decides whether to inspect each merchant's bag. Merchants can bribe the Sheriff or try to convince them they're honest. After 4 rounds (each player serves as Sheriff once), the richest player wins.

## Players
- Number of players: 4 (Alice, Bob, Charlie, Diana)
- Roles: All players share the "Player" role. The Sheriff role rotates — each player is Sheriff exactly once across 4 rounds.

## Objective
Maximize your total gold after 4 rounds. Players in the top half by gold win; players in the bottom half lose.

## How to Play
Each round, one player is designated Sheriff. The other 3 are Merchants.

**Phase 1 — Pack (private, simultaneous):**
- Each Merchant secretly chooses to pack `honest` (3 legal goods worth 3 gold) or `smuggle` (contraband worth 8 gold if not caught), replying with exactly one of the two words.
- A longer reply is read by the choice it commits to (`committed_move.py`, shared by the bluffing games): "I'd rather smuggle than be honest", "Honestly, I'll smuggle" and "To be honest, smuggling pays more: smuggle" are smuggle; "smuggling" and "contraband" count as smuggle, "pack honestly" as honest, while "honestly" and "to be honest" as figures of speech, and what the merchant wants the Sheriff to believe ("I'll convince her I'm honest"), are no choice. With no committed choice, a negated one counts as the other ("I am not going to smuggle" is honest). A reply that names neither, or commits to both, packs honest goods. Either way the merchant is told privately how the reply was read (a public notice would reveal the bag).

**Phase 2 — Negotiate (public, round-robin speaking):**
- The Merchants speak in seat order starting with the player after the Sheriff (clockwise from the Sheriff's left, as in the rulebook); the Sheriff speaks last, after hearing them.
- Merchants can offer bribes, make promises, or argue they packed honestly.
- The Sheriff can demand bribes, threaten inspections, or accept deals.
- A bribe is binding when the Merchant offers, in the first person, to give or pay the Sheriff N gold in this round, in any wording, as an immediate bribe is in the rulebook (owner's decision, third round): "I offer you 3 gold", "I'll give you 2 gold", "I'd like to offer 2 gold", "I would offer you 2 gold", "my offer is 3 gold", "a bribe of 2 gold", "I'll give you 2 if you pass me"; hypothetical, conditional and question forms bind too. Amounts may be digits or words up to twenty, with "gold", "coins" or "g". If the Sheriff passes that Merchant, N gold goes from the Merchant to the Sheriff, paid after the Merchant's goods and capped at what the Merchant then holds. An inspected Merchant pays nothing. The offer is announced to everyone. An offer does not bind if it is negated ("I won't pay you 2 gold"), made by someone else ("Bob offers you 2 gold", "Charlie's offer of 3 gold", "Jonathan offering 1 gold looks suspicious"), addressed to another player ("I'll give Bob 2 gold"), or explicitly for a later round ("next round I'll give you 2 gold", "when I'm the Sheriff"). If a statement makes several binding offers, the last one counts. Promises of anything other than gold this round are not enforced. After its statement each Merchant is told privately how it was read: a binding offer of N gold, or no binding offer (also when the model call failed).

**Phase 3 — Inspect (simultaneous):**
- The Sheriff decides for each Merchant: `inspect NAME` or `pass NAME`. A verb the reply commits to applies to the list of names right after it ("I will inspect Bob, Charlie and Diana", "Inspect: Bob, Charlie. Pass: Diana.", "I'm inspecting Bob and passing Diana"), or to every Merchant ("inspect all merchants"); "Bob: inspect" names the Merchant first; a negated or hypothetical verb decides nothing. The instruction lists the Merchants clockwise from the Sheriff's left with no example decision, and shows the binding offers. A Merchant with no decision, or with both decisions, is passed, and the engine says so.
  - **Inspected + honest:** Sheriff pays a 2-gold penalty to the Merchant.
  - **Inspected + smuggling:** Merchant pays a 4-gold penalty to Sheriff and earns 0 from goods.
  - **Passed:** Merchant keeps their goods (honest=3 gold, smuggle=8 gold).

Gold can go negative (a caught smuggler or a Sheriff who inspects honest Merchants may drop below 0). After 4 rounds, total gold determines the winner.

## Scoring / Payoffs
After 4 rounds, players are ranked by total gold. The top half win; the bottom half lose.
- Top half by gold (2 of 4 players): +1.0 reward
- Bottom half by gold (2 of 4 players): -1.0 reward
- Players tied on gold share the mean of the results of the places they span, so the rewards always sum to zero: two tied across the middle get 0 each, three tied for first get +1/3 each (the fourth -1), three tied for last get -1/3 each (the first +1). All tied: draw for everyone. Seating order never decides. The rules text states this.

## Social Skills Tested
- **Bluffing:** Smugglers must convince the Sheriff they packed honestly.
- **Negotiation:** Bribing and deal-making during the Negotiate phase.
- **Trust calibration:** The Sheriff must estimate which merchants are lying based on behavior and history.
- **Risk assessment:** Merchants must decide when smuggling is worth the inspection risk given who is Sheriff.
