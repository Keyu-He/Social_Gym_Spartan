# Coup

## Overview
Coup is a compact bluffing game of hidden roles. Each player has 2 secret influence cards and 2 coins. On your turn, you declare an action — some actions claim a specific card role (which you may or may not actually have). Other players can challenge your claim (if they think you're bluffing) or block you (if they claim a counter-role). Getting caught in a bluff costs you an influence card (life). Lose both cards and you're eliminated. Last player standing wins.

## Players
- Number of players: 4 (Alice, Bob, Charlie, Diana)
- Roles: All players are "Player" roles; card identities are hidden. Each player holds 2 influence cards.

## Objective
Be the last player with at least one influence card remaining. Bluff strategically, challenge wisely, and eliminate opponents.

## How to Play
Each turn, the active player declares an action. Turns go clockwise (seat order) among the players still in the game. Other players respond:

**Actions you can take:**
- `income`: Take 1 coin. Cannot be challenged or blocked.
- `foreign_aid`: Take 2 coins. Claims no role, so it cannot be challenged. Any other player can block it by claiming Duke.
- `tax`: Claim Duke, take 3 coins. Can be challenged.
- `steal NAME`: Claim Captain, take 2 coins from NAME. Can be challenged; only NAME can block it, by claiming Captain or Ambassador.
- `assassinate NAME`: Claim Assassin, pay 3 coins, NAME loses a life. Can be challenged; only NAME can block it, by claiming Contessa. The 3 coins are paid on declaring: a successful challenge returns them, a block does not.
- `exchange`: Claim Ambassador, draw 2 cards, then keep as many of your cards plus the 2 drawn as you have influence (`keep CARD` or `keep CARD1 CARD2`). Can be challenged. The cards after the keep word the reply commits to count ("Return Captain and Ambassador; keep Duke and Contessa" keeps Duke and Contessa; quoted or bold names and plurals are names); a reply that only says what it returns keeps the rest. A reply that does not name exactly the right number of the offered cards, or no reply, keeps the original hand, said publicly; the rules text states this.
- `coup NAME`: Pay 7 coins, NAME loses a life. Cannot be challenged or blocked. **Mandatory if you have 10+ coins.**

**Response phase:**
- `challenge`: You think the declarer doesn't have the claimed card. If right, declarer loses a life; if wrong, you lose a life. Only role claims can be challenged (not foreign aid).
- `block`: Claim a counter-role to block the action. Anyone can block foreign aid; only the target can block a steal or an assassination; tax and exchange cannot be blocked. A block of a steal may name its role (`block Captain` or `block Ambassador`); one naming no role claims every allowed role. A block whose claimed roles (the card names the reply commits to, not another player's cards) include none that can block the action, such as "I block the steal with my Contessa", is announced as an invalid block and counts as no block. The claim is not corrected to an allowed role.
- `pass`: Let the action go through.

Only the answer a reply commits to counts (`committed_move.py`, shared by the bluffing games): "pass, I won't challenge", "I don't think it's worth challenging here, so I pass" and "Challenging would be too risky. Pass." are passes. A challenge or block that the action does not allow, a reply that commits to two different answers, any other answer, and a missing answer all count as a pass, with a public notice. The action word of a declaration may stand anywhere in the reply ("I declare income, taking 1 coin."); an action the reply does not commit to does not count ("income now, coup Bob later" is income, "Since nobody can block tax, tax" is tax). A declaration the engine cannot carry out (no action or more than one named, an action word inside block or challenge wording such as "block Bob's assassinate with Contessa", unaffordable, no living target, or not a coup at 10+ coins) is rejected publicly and the declarer loses the turn; nobody is asked to respond.

**Order of resolution (official rules):** a challenge of the action's claim is settled first. If it fails (the challenger loses an influence and the declarer swaps the card it showed for a new one), or nobody challenged, a block given in the same response step stands. A target that challenged a steal or an assassination answered before its challenge was settled, so once its challenge fails it is asked in a separate Block step whether it blocks (`block` or `pass`), as the rulebook allows. When several players challenge (or block) in the same step, the first one clockwise from the declarer does; when several challenge a block, the declarer does if it is among them, else the first one clockwise from it. Every block can then be challenged by any player other than the blocker, in a separate step: if the blocker has the claimed card, the challenger loses an influence and the action is cancelled; if not, the blocker loses an influence and the action takes effect. An unchallenged block cancels the action. A blocked assassination's 3 coins stay spent.

**Losing an influence:** the player chooses which card to reveal (`lose CARD`, asked privately). A player with one card, or two of a kind, loses one without being asked. A reply that names none of the player's cards, two different cards to lose, or no reply, loses a card drawn at random (seeded by the roster), announced publicly. "I won't lose my Duke" keeps the Duke; a reply that opens with a card name ("Captain, since the Duke is more useful") loses that card.

**Card types:** Duke, Assassin, Captain, Ambassador, Contessa (3 copies each in 15-card deck)

## Scoring / Payoffs
- Winner (last player standing): +1.0
- Eliminated players: -1/3 each (zero-sum)

## Social Skills Tested
- **Bluffing:** Claiming cards you don't have to access powerful actions.
- **Challenge timing:** Knowing when the odds favor challenging vs. letting an action slide.
- **Resource management:** Building coins toward coup while managing your influence cards.
- **Threat assessment:** Prioritizing which opponents to eliminate and when.
