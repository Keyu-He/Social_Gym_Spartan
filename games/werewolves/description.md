# Werewolves (Mafia)

## Overview
Werewolves is a classic social deduction game of hidden roles. Each night, the Werewolves secretly eliminate a Villager. Each day, all players discuss and vote to execute someone they suspect is a Werewolf. Special roles (Seer, Witch) have unique night abilities. The game alternates between night and day until one side wins.

## Players
- Number of players: 6 total
  - 2 Werewolves (know each other's identity)
  - 2 Villagers (no special abilities)
  - 1 Seer (can inspect one player per night to learn their role)
  - 1 Witch (has one save potion and one poison potion, usable once each)

## Objective
- **Villagers/Seer/Witch:** Eliminate all Werewolves through day votes.
- **Werewolves:** Kill enough Villagers at night so that Werewolves equal or outnumber the remaining Villagers.

## How to Play
The game alternates between Night and Day cycles:

**Night phase (in order):**
1. **Werewolves:** Secretly choose one player to kill (`kill NAME`). Any living player may be chosen, a Werewolf included (naming a partner or oneself is allowed, for example to draw out the Witch's save; the models are told they may choose any living player). The Werewolves propose one after the other in seat order; if they name different players, one of the two is drawn at random (seeded by the roster) as the victim, and the models are told this rule. Only Werewolves see each other's actions.
2. **Seer:** Privately inspect one player (`inspect NAME`) to learn if they are a Werewolf or Villager.
3. **Witch:** May use save potion (`save NAME`) on tonight's victim, or poison potion (`poison NAME`) on anyone. Each potion is one-time use, and the Witch may use at most one potion per night; the rules text and the Witch's instruction say so. This follows the common Chinese ruleset (the owner's choice); Miller's Hollow lets the Witch use both potions on the same night. A save on anyone other than tonight's victim has no effect, the potion is kept, and the Witch is told privately whom the save would work on.

Night actions only work on living players (every prompt lists them). A night action is read like a vote: the name right after "kill", "inspect", "save" or "poison", or else the only player the reply names (the Witch must use a potion command); a negated command or mention does not count ("Don't kill Erin, kill Finn" is a kill on Finn, "I won't poison anyone, save Erin" is a save). A reply with no usable move, a failed model call, a potion command for a used potion, and a second potion command on the same night (one potion per night) each get a private notice to the acting player; a Witch's "skip" is a silent pass. A kill, inspect, save or poison that names a dead player, or no player, has no effect (a potion is not used up), and the acting player is told so privately (the Werewolves as a team). The Werewolf phase always takes one turn per Werewolf dealt, in seat order, with an idle turn for a dead Werewolf, so the length of the night does not reveal how many Werewolves are left.

**Day phase (in order):**
4. **Dawn:** All players learn who died during the night. A dead player's role is never revealed, at night or at an execution; the rules text says so. This follows the common Chinese ruleset (the owner's choice); in Miller's Hollow a dead player's card is turned face up.
5. **Discussion:** All living players speak publicly, sharing suspicions and debating who is a Werewolf.
6. **Vote:** Each player votes to execute one suspect (`vote NAME`). Most votes = executed; a tied top vote is settled by a seeded random draw, announced to everyone (the models are told this rule). A vote counts for the name right after "vote", or else for the only player the reply names; a negated mention does not count ("I won't vote Bob, vote Carol" is a vote for Carol). The game lists how each vote was read. A vote for a dead player, or one that names no player, counts as an abstention and the game says so to everyone; if no vote counts, nobody is executed.
7. **Twilight:** Execution result announced. Return to Night.

## Scoring / Payoffs
The winning side splits +1.0 and the losing side splits -1.0, so with 2 Werewolves and 4 village players (2 Villagers, the Seer and the Witch):
- Village wins (all Werewolves eliminated): each village player +1/4, each Werewolf -1/2
- Werewolves win (Werewolves ≥ remaining village players): each Werewolf +1/2, each village player -1/4

A 40-turn cap ends an undecided game as a draw (every player 0). It is a backstop against a game that never ends, so the models are not told about it.

## Social Skills Tested
- **Deception:** Werewolves must argue convincingly that they are Villagers.
- **Accusation and defense:** Players must justify suspicions and counter accusations.
- **Information use:** The Seer must decide how to use their private knowledge without outing themselves to Werewolves.
- **Social influence:** Persuading others to vote for (or against) a specific target.
- **Theory of mind:** Modeling what each player knows and what their behavior reveals.
