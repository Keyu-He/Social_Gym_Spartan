# Bohnanza

## Overview
Bohnanza (Uwe Rosenberg, AMIGO 1997) is a trading and set-collection card game. Every
player is a bean farmer: you plant beans in your fields, harvest a field for gold coins
once it holds enough beans of one variety, and, because your hand is held in a fixed order
that you cannot rearrange, you must trade constantly to avoid planting beans you do not
want. The winner is the player with the most coins.

## Players
- 4 players (the benchmark configuration; the published game plays 3 to 5).
- Everyone has the same role. Hands are private; fields, coins and hand sizes are public.

## Cards
The published deck has 154 bean cards of 11 varieties; with 4 or 5 players the rules
remove the Coffee Beans, so the benchmark plays 130 cards of 10 varieties
(`removed_varieties` in config.json; set it to `[]` for the full deck). Each variety's
**beanometer** gives the number of beans that must be harvested at once to earn
1 / 2 / 3 / 4 coins (a dash means that level does not exist).

| Variety | Cards | 1 coin | 2 coins | 3 coins | 4 coins |
|---|---|---|---|---|---|
| Coffee | 24 | 4 | 7 | 10 | 12 |
| Wax | 22 | 4 | 7 | 9 | 11 |
| Blue | 20 | 4 | 6 | 8 | 10 |
| Chili | 18 | 3 | 6 | 8 | 9 |
| Stink | 16 | 3 | 5 | 7 | 8 |
| Green | 14 | 3 | 5 | 6 | 7 |
| Soy | 12 | 2 | 4 | 6 | 7 |
| Black-eyed | 10 | 2 | 4 | 5 | 6 |
| Red | 8 | 2 | 3 | 4 | 5 |
| Garden | 6 | - | 2 | 3 | - |
| Cocoa | 4 | - | 2 | 3 | 4 |

## Setup
Each player gets 2 bean fields and 5 cards, dealt one at a time into a hand whose order
is fixed: new cards go to the back, only the front card can be planted. The rest of the
deck is the draw pile.

## A turn (players take turns in seating order)
1. **Plant.** The active player must plant the front card of their hand and may also
   plant the next card. A field holds one variety only; planting a different variety
   into an occupied field means harvesting that field first.
2. **Trade.** Two cards are turned face up from the draw pile for the active player.
   The active player may trade those two cards and any hand cards; the other players
   may trade cards from their hands, but only with the active player. Gifts (a card
   for nothing) are trades and need the receiver's consent. Cards received in a trade
   may not be traded on: they are set aside and must be planted in step 3.
3. **Plant traded cards.** Every player plants the cards they received. The active player
   also plants any face-up card that was not traded away.
4. **Draw.** The active player draws 3 cards, one at a time, to the back of their hand.

**Harvest and the third field.** A player may harvest any of their fields, and may buy a
third field for 3 coins (once), at any time: in the engine, in any message they send
(Plant, Trade, Resolve, Plant traded). Harvest: count the beans in the field, take coins
per the beanometer (the coin cards leave the game), discard the rest; coins spent on a
field go back to the discard pile as cards. **Bean protection:** a field holding a single
bean may not be harvested while another of your fields holds 2 or more.

## Game end and scoring
The game ends after `max_rounds` (default 6) full rounds, or when the draw pile runs out:
if it runs out while turning up the face-up cards, the turn is played to the end of the
planting step and the game ends without a draw; if it runs out during the draw of three,
the game ends at once (cards already drawn stay in the hand). Everyone then harvests
every field.
Coins decide the ranking, equal coin totals are ordered by the number of cards left in
hand (the Rio Grande tie-break). Every place has its own reward, evenly spaced (the rules
text says so): 1st +1,
2nd +1/3, 3rd -1/3, 4th -1 (`graded_rank_rewards`); players still equal share the mean
of the places they span. The Elo fit only compares rewards pairwise, so this gives six decisive
comparisons per game instead of the four-plus-two-ties of the top-half/bottom-half rule
used by Sheriff.

## Engine protocol
Messages are free text; the engine reads commands out of them. Before parsing it drops
`<think>` blocks, markdown and step labels, and writes field ordinals as digits ("the
second field" is field 2). A message is cut into fragments at punctuation and before each
verb; a fragment counts as a command only if its verb is not negated or hedged in the
words in front of it ("I don't want to harvest 1", "I plan to harvest 1 later" do
nothing). When prose and a bare command both appear ("I will plant my first card into
field 1. plant 1"), the bare command wins. Rejected or unreadable commands are answered
with a private notice to the sender, never broadcast, and counted (`rejected`).

The per-turn instructions give command formats with placeholders (`plant <field number>`,
`plant <card> in field <number>`, `offer <Name>: give <cards> for <cards>`, `accept #<id>`)
and never a filled-in move: small models copy a filled-in example verbatim, and the old
examples ("plant 1, plant 2", card i in field i, an offer for Soy to the next seat) uprooted
fields or asked for cards the player had no field for. A format copied without filling it
in plants into the matching or an empty field (as a plant without a field number does),
accepts the only open offer (as a bare `accept` does), or posts no offer and gets the
"No offer was posted" notice.

- **Plant** (active player, one message). `plant <field>` (also `put`/`place`/`sow`) plants
  the current front card; a second `plant` plants the next card (at most two; `plant 1 and
  2`, `plant both` also work). A field number is read from `field N` / `#N`, else from a
  bare digit after bean names and quantities are removed. If no legal plant is found, the
  engine plants the front card itself: matching field, else empty field, else after
  harvesting the legal field that pays most (`fallbacks` counter, public notice). The rules
  text states this. A command that names a card other than the front card (`plant Soy in
  field 2` with Chili in front) gets a private notice and plants the front card into the
  field it names when that is legal (here harvesting the field first, as any plant into an
  occupied field does), else the usual placement applies; the engine does not move the card
  to a field it guesses was meant.
- **Trade** (round-robin, active player first, one message each). Offers: `offer <Name>:
  give <cards> for <cards>`, and also `offer Bob Cocoa for Soy`, `trade X for Y with Bob`,
  `give Bob a Cocoa`, `X -> Y`; `nothing` on either side makes a gift. Sides never cross a
  sentence boundary and never contain another `offer`, so a lead-in such as "I'm
  offering:" does not swallow the offer on the next line (`offering` is one word, never a
  player called "ing"); an `or` on either side rejects the offer with a notice. The active
  player must name a player; others' offers go to the active player. An offer is posted
  only if the proposer holds the cards they give (told privately otherwise); whether the
  other side holds the requested cards is checked at acceptance and told privately, so an
  offer never reveals a hidden hand. Accepting: `accept #<id>`, `accept 1 and 3`,
  `accept #1-#3`, `accept all`, `accept all except #2`, `accept Bob's offer`, `accept the
  first offer`; a bare `accept` / `yes` takes the only open offer or asks for the id.
  Only an accept the message commits to counts (`committed_move.py`, shared by the
  bluffing games): negated, hypothetical, conditional and later forms (`I do not accept #1`,
  `I would accept 2 Soy for a Chili`, `I'd accept if you add a Soy`, `Maybe I'll accept #1
  later`) and declines in words (`I accept none of these offers`, `Accept nothing`, `I
  accept neither offer`, `I accept no offers`) accept nothing. An accepted trade executes at once; offers whose given cards were traded
  away are void.
- **Resolve** (active player only, skipped when nobody made them an offer): the accept
  forms above, `pass`, or anything else, which declines everything (announced; a failed
  model call is announced the same way).
- **Plant traded** (only players holding received or unclaimed face-up cards, one message):
  `plant <Bean> in field <n>` per card; `Soy 1`, `field 1: Soy` also work; anything left is
  placed automatically as in Plant.
- `harvest <field>` and `buy field` are accepted in any message from the player who sends
  it, but only as verb-first, non-negated commands.
- The turn cap (400 turns, never reached in play) ends the game with the current standings.

## Original rules and benchmark adaptations
Deck composition (including the 4-player Coffee removal), beanometers, hand order, the
turn structure, trading restricted to the active player, consent for gifts, the
no-re-trade rule, bean protection, harvesting at any time, the third field and its price
follow the Rio Grande / AMIGO 154-card rule booklet. Adaptations:
- **Length.** The published game ends the third time the draw pile runs out; here it ends
  the first time, or after 6 rounds (`deck_exhaustions_to_end`, `max_rounds` in
  config.json), to keep an episode near 100 model calls. With 130 cards and 4 players the
  pile runs out during round 6, so both limits matter.
- **One pass of trading per turn.** Each player speaks once and the active player answers
  once, instead of open-ended table talk. The active player's own proposals go in their
  opening message and are answered by the others in theirs. "Any time" harvesting and
  field purchases are therefore possible once per message a player sends.
- **Tie-break.** The Rio Grande booklet breaks ties by cards in hand (used here); the AMIGO
  edition uses seating distance, which the benchmark never lets decide a game, so players
  equal on both counts share a fate.
- **Coins spent on a field** return to the discard pile as cards, as in the booklet; the
  variety printed on a coin card is irrelevant, so the engine returns cards of the most
  common variety in play (Wax). With `deck_exhaustions_to_end` at 1 the discard pile is never
  reshuffled, so these cards never come back into play.
- Harvests at game end ignore bean protection, as in the rulebook's final harvest.
- Timeouts: an episode is cut at 400 turns by the evaluator (never reached in tests).

## Social skills tested
- **Negotiation:** proposing exchanges, valuing cards by what the counterpart must plant.
- **Theory of mind:** reading which beans the others are collecting from their fields.
- **Strategic reasoning:** when to harvest, when to accept a bad card to protect a field.
