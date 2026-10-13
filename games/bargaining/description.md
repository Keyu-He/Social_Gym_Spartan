# Bargaining (Iterated Ultimatum Game)

## Overview
The Bargaining game is an iterated ultimatum game. Each round, one player (the Proposer) splits 10 tokens between themselves and the other player. The other player (the Responder) either accepts the split (both get their shares) or rejects it (both get nothing). Roles alternate each round. This tests fairness norms, negotiation, and strategic offer-making.

## Players
- Number of players: 2
- Roles: Both players alternate between "Proposer" and "Responder" each round.

## Objective
Maximize your own total tokens over 10 rounds. Higher total score wins.

## How to Play
1. Each round, the Proposer uses `offer N` where N is how many tokens the Proposer keeps (Responder gets 10-N).
2. The Responder then chooses `accept` or `reject`.
   - Accept: both get their proposed shares.
   - Reject: both get 0 for that round.
3. Roles switch each round (if Alice proposes in round 1, Bob proposes in round 2).
   - The proposal is read from `offer N`, which always counts the tokens the Proposer keeps; "I keep 6" and "you get 4" are read for what they say, a bare number counts as `offer N`, and a negated form ("I won't offer 3") does not count. A proposal is unclear when it has no number, a number outside 0 to 10, readings that disagree ("offer 4, you get 4", and so "offer 4 to you" and "I offer 4 tokens to you", where `offer 4` keeps 4 and "4 to you" gives 4), or the word accept or reject (the Responder's move, which 7 logged proposals gave).
   - The response is the decision the reply makes: every decision word counts (accept, acceptable, reject, decline, refuse, unacceptable), a negated one as the other ("I do not accept" rejects); words that only describe a decision ("accepting", "rejection") do not. A reply that makes no decision, or both, is unclear.
   - An unclear reply or a failed model call is announced publicly and the same player is asked once more. If the second reply is also unclear, the offer is drawn uniformly from 0 to 10, or the response is accept or reject with equal chance, from a stream seeded by the roster; the round line marks the draw.
   - Both phases are private: the other player sees only the parsed offer ("Alice offers to keep 6; Bob would get 4") and the round result, never the reply's text. Each player keeps its own replies in its history.
4. After 10 rounds, the higher-scoring player wins.

## Scoring / Payoffs
Per round:
- If accepted: Proposer gets N tokens, Responder gets 10-N tokens
- If rejected: Both get 0

Win conditions:
- Higher scorer wins (+1.0), lower loses (-1.0)
- Tied scores: Draw (0.0 each); the rules text says so

## Social Skills Tested
- **Negotiation:** Making offers that are fair enough to be accepted while maximizing personal gain.
- **Fairness norms:** Deciding when to reject "unfair" offers even at personal cost.
- **Role-switching strategy:** Adapting behavior as both Proposer and Responder across different rounds.
- **Credible threats:** Whether a reputation for rejecting unfair offers influences future proposals.
