# Public Goods Game

## Overview
The Public Goods Game models the free-rider problem. Each player starts each round with 10 tokens and secretly decides how many to contribute to a shared pool. The pool is multiplied (making collective contribution beneficial) and split equally among all players. Players who contribute little keep more tokens but benefit from others' contributions. After each round, individual contributions are revealed.

## Players
- Number of players: 4
- Roles: All players are symmetric "Player" roles.

## Objective
Maximize your own total earnings over 10 rounds relative to other players. The top 2 earners win; the bottom 2 lose. Relative ranking matters more than absolute earnings.

## How to Play
1. Each round, all 4 players simultaneously and secretly choose how many tokens (0–10) to contribute to the shared pool.
2. Contributions are summed, multiplied by 1.5, then divided equally among all 4 players.
3. Each player's round earnings = (tokens kept) + (pool share).
4. After each round, all individual contributions are revealed publicly.
5. After 10 rounds, final earnings determine the ranking.

## Scoring / Payoffs
Per round (example):
- If all 4 contribute 10: pool = 40 × 1.5 = 60, each gets 60/4 = 15
- If you contribute 0, others each contribute 10: pool = 30 × 1.5 = 45, you get 10 + 11.25 = 21.25, others get 0 + 11.25 = 11.25

Win conditions:
- Top 2 earners: +1.0 reward
- Bottom 2 earners: -1.0 reward
- Players tied on earnings share one result: a tied group inside the top half wins, inside the bottom half loses, and one straddling the boundary draws (0.0). All tied: draw for everyone. Seating order never decides. The description, the goal and every turn's instruction state this rule, with two worked examples (30, 30, 30, 12: three draws and a loss; 30, 20, 20, 12: a win, two draws and a loss).

Earnings are kept exactly (shares are eighths of a token); messages round them to 3 decimals only for display, so equal true earnings always tie.

## Reading a contribution
The contribution is the number after `contribute` ("In round 3 I contribute 7" is 7); a reply without `contribute` must hold exactly one number. A negated amount ("I won't contribute 5") does not count. Two different amounts, no amount, a fraction of a token, or an amount below 0 or above 10 is unclear; out-of-range amounts are asked again, not clamped.

## Unclear replies
A reply that does not give one whole number of tokens (including a failed model call) is not read as a contribution. The engine says so to everyone and asks that player once more; every other contribution stays hidden, because replies are private. If the second reply is also unclear, the contribution is drawn at random from 0 to 10, each amount equally likely, from a stream seeded by the roster; a failure never picks a fixed amount such as 0, the rank-dominant move. The round results mark a drawn contribution. The match still lasts 10 rounds; a round with a repeated reply takes one extra step.

## Social Skills Tested
- **Free-riding:** Benefiting from others' contributions while contributing little yourself.
- **Punishment and reputation:** Whether players punish low contributors in subsequent rounds by reducing their own contributions.
- **Social norms:** Maintaining cooperative norms under the temptation to free-ride.
- **Strategic contribution:** Contributing just enough to appear cooperative while maximizing personal gain.
