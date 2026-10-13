# Insider

## Overview
Insider is a cooperative deduction game with a hidden saboteur. The whole group tries to guess a secret word by asking yes/no questions, but one player — the Insider — secretly knows the word and must subtly guide the group toward guessing it. If the group succeeds, a new challenge begins: identify who was helping too much (the Insider). The Insider wins by hiding their identity after the word is guessed.

## Players
- Number of players: 5 total
  - 1 Insider (secretly knows the word, must guide without being obvious)
  - 4 Citizens (don't know the word, ask yes/no questions to figure it out)

## Objective
- **Citizens:** Guess the secret word by asking smart yes/no questions, then identify who the Insider is and vote them out.
- **Insider:** Help the group guess the word subtly, then avoid being identified in the subsequent vote.

## How to Play
1. **Questioning phase:** Players take turns asking yes/no questions (15 in total; every turn counts as one, and a turn with no question is announced as such). The environment answers each question in place of the Master, with Yes, No, or I don't know. A question that asks directly whether the answer is the secret word (or a plural, hyphen or space variant of it) is answered Yes by rule; every other question goes to an LLM judge that is told the word, its category and the episode's earlier questions and answers. The judge is `gpt-6-luna` through the OpenAI API (a reasoning model; no temperature and no token limit are sent), set in `config.json` (`judge_model`, `judge_params`); `SOTOPIA_INSIDER_JUDGE`, `SOTOPIA_INSIDER_JUDGE_BASE_URL`, `SOTOPIA_INSIDER_JUDGE_API_KEY` and `SOTOPIA_INSIDER_JUDGE_PARAMS` (a JSON object of request fields) select another one, for example Qwen3.8-27B on vLLM with `SOTOPIA_INSIDER_JUDGE_PARAMS='{"max_tokens": 4096, "temperature": 0, "extra_body": {"chat_template_kwargs": {"enable_thinking": true}}}'`. Only the judge's final answer is read: a reasoning block is dropped, and a reply that is not Yes, No or I don't know is asked again once and then shown as an unreadable answer ("I don't know" with a note), as is a failed call. The Insider should subtly steer questions toward the answer. The Insider's prompt names the word from the start, and the first questioner's instruction names the category.
2. **Guess phase:** All players simultaneously guess the word (`guess WORD`) or pass. A guess names the word if its first clause that says something, without lead-ins such as "guess", "I think the word is", "It's" or "My guess:" and without quotes or an article, is the word or a plural, hyphen or space variant (the same variants the question rule accepts); a negated part does not count ("It's not a koala, it's a kangaroo" guesses kangaroo). A failed or empty reply is announced and counts as a pass.
   - If no one guesses correctly: everyone loses.
   - If someone guesses correctly: proceed to Discussion.
3. **Discussion phase:** Players discuss who they think the Insider is based on who asked suspiciously helpful questions.
4. **Vote phase:** All players simultaneously vote for the Insider (`vote NAME`). Plurality vote determines the result; a tied top vote is settled by a seeded random draw, announced to everyone (the models are told this rule). A vote counts for the name right after "vote", or else for the only player the reply names; a negated mention does not count ("I won't vote Bob, vote Carol" is a vote for Carol). The result line lists how each vote was read. A vote that names no single player is an announced abstention, with its reason, and the vote still resolves on that turn; if nobody casts a valid vote, the Insider is not identified and wins.

## Scoring / Payoffs
When the word is guessed, the winning side splits +1.0 and the losing side splits -1.0, so with 1 Insider and 4 Citizens:
- Word NOT guessed: everyone loses, every player -1.0 (not zero-sum)
- Word guessed + Insider identified (plurality vote): each Citizen +0.25, the Insider -1.0
- Word guessed + Insider NOT identified: the Insider +1.0, each Citizen -0.25

## Social Skills Tested
- **Subtle guidance:** The Insider must help without being too obvious about knowing the answer.
- **Deduction:** Citizens must narrow down the word with efficient yes/no questions.
- **Behavioral inference:** Citizens must notice who asked suspiciously leading questions.
- **Concealment:** The Insider must ask questions that look like genuine guesses, not insider knowledge.
