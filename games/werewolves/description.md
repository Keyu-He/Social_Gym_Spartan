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
1. **Werewolves:** Secretly choose one player to kill (`kill NAME`). Only Werewolves see each other's actions.
2. **Seer:** Privately inspect one player (`inspect NAME`) to learn if they are a Werewolf or Villager.
3. **Witch:** May use save potion (`save NAME`) on tonight's victim, or poison potion (`poison NAME`) on anyone. Each potion is one-time use.

**Day phase (in order):**
4. **Dawn:** All players learn who died during the night.
5. **Discussion:** All living players speak publicly, sharing suspicions and debating who is a Werewolf.
6. **Vote:** Each player votes to execute one suspect (`vote NAME`). Most votes = executed.
7. **Twilight:** Execution result announced. Return to Night.

## Scoring / Payoffs
- Villager team wins (all Werewolves eliminated): +1.0 for Villagers/Seer/Witch, -1.0 for Werewolves
- Werewolf team wins (Werewolves ≥ remaining Villagers): +1.0 for Werewolves, -1.0 for Villagers/Seer/Witch

## Social Skills Tested
- **Deception:** Werewolves must argue convincingly that they are Villagers.
- **Accusation and defense:** Players must justify suspicions and counter accusations.
- **Information use:** The Seer must decide how to use their private knowledge without outing themselves to Werewolves.
- **Social influence:** Persuading others to vote for (or against) a specific target.
- **Theory of mind:** Modeling what each player knows and what their behavior reveals.
