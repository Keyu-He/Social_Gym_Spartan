"""Read the move a free-text reply commits to.

Shared by the bluffing games (Liar's Dice, Skull, Coup, Sheriff of Nottingham, Bohnanza). The game modules are
loaded by file path, not as a package, so each loads this file by path too (`_shared` in its main.py).

A reply often names moves it does not make: "I could bid 5, but I'll pass", "I'm not going to call liar; bid 4 5",
"Honestly, I'll smuggle", "I don't think it's worth challenging, so I pass". Each game lists its move words as
regular expressions, and a mention counts only when the reply commits to it. It does not count when its own clause
negates it, makes it hypothetical or conditional, asks about it, puts it on a later turn or in the past, or talks
about it as the subject of a sentence ("Challenging would be too risky", "Foreign aid could be blocked"). A reply
that commits to two different moves is unclear; each game sends it down its declared unclear-reply path.

A clause ends at sentence punctuation, at a comma or dash, and before a linking word ("but", "so", "because",
"since", ...); "and" ends one before "I", "we" or "then", and directly before a move word ("I won't call liar and
bid 4 5" commits to the bid).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional, Pattern, Tuple, Union

COMMITTED = "committed"


def normalize(text: str) -> str:
    """Lower case, straight quotes, and markdown and quotation marks blanked out ("**keep 'Duke'**" is read as
    "keep Duke"). Every character keeps its position, so a match in the result indexes the original text too."""
    t = (text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = "".join(c.lower() if len(c.lower()) == 1 else c for c in t)
    t = re.sub(r"[*_`#~\"]", " ", t)
    # a quotation mark around a word, not an apostrophe inside one ("I'll", "Bob's")
    return re.sub(r"(?<![a-z0-9])'|'(?![a-z])", " ", t)


# ---------------------------------------------------------------------------------------------------------------
# Clause boundaries
# ---------------------------------------------------------------------------------------------------------------

_HARD = re.compile(r"[!?;:\n()\[\]]|\.(?!\d)|\s[-\u2013\u2014]+\s|[\u2013\u2014]")
_SOFT = re.compile(
    r",|\b(?:but|yet|however|though|although|because|cause|since|therefore|thus|hence|whereas|while|"
    r"instead(?!\s+of)|so(?!\s+(?:far|much|many|long|that|as|be)\b))\b"
    r"|\band(?=\s+(?:i|i'll|i'm|i'd|i've|we|we'll|let's|let|then|now)\b)"
)
_AND_BEFORE = re.compile(r"\band\s+(?:(?:the|a|an|my)\s+)?$")

# ---------------------------------------------------------------------------------------------------------------
# Markers. Left: between the clause start and the mention. Right: from the mention to the clause end.
# ---------------------------------------------------------------------------------------------------------------

_NEGATION = re.compile(
    r"\b(?:not|no|never|none|nothing|nobody|neither|nor|without|cannot|refus\w*|declin\w*|avoid\w*|against|"
    r"forgo\w*|instead of|rather than|other than|than|no one|hold(?:ing)? off|pass(?:ing)? on|skip(?:ping)?|"
    r"refrain(?:ing)? from)\b|n't\b"
)
_NEGATIVE_OBJECT = re.compile(r"^\s*(?:none|nothing|neither|no|nobody|no one|zero)\b")
_HYPOTHETICAL = re.compile(
    r"\b(?:could|might(?!\s+as\s+well)|maybe|perhaps|possibly|if|unless|whether|suppose|supposing|assuming|"
    r"imagine|hypothetically|consider\w*|tempt\w*|alternatively|otherwise|wonder\w*|risky|dangerous|too|"
    r"should i|shall i|do i|ought i|can i|what if|how about|what about|why not|think(?:ing)? (?:about|of)|"
    r"thought (?:about|of))\b|(?:\bwould|'d)\b(?!\s+(?:rather|prefer|like|love|better|be happy|be glad))"
)
_CONDITIONAL_AFTER = re.compile(r"\b(?:if|unless|provided|as long as|in case|assuming)\b")
_LATER = re.compile(
    r"\b(?:later|next (?:turn|round|time|move|one)|following (?:turn|round)|afterwards?|after that|eventually|"
    r"subsequently|in the future|future|some ?day|down the line|at some point|another (?:turn|round|time)|"
    r"and then)\b"
)
_PAST_BEFORE = re.compile(r"\b(?:was|were|been|used to|previously|earlier|already|last (?:turn|round|time))\b")
_PAST_AFTER = re.compile(
    r"\b(?:last (?:turn|round|time)|so far|until now|up to now|earlier|previously|ago|in (?:the )?past)\b"
)
# A move named as the subject of a sentence about it: a gerund subject with a verb within three words after it
# ("Challenging would be too risky", "Blocking needs a Contessa", "being honest doesn't pay", "Calling liar on Bob
# is risky"), or the move word followed at once by a modal or a negated verb ("Foreign aid could be blocked").
_FINITE = re.compile(
    r"^\s*(?:\S+\s+){0,2}?(?:is|was|are|were|would|could|might|may|can|will|won't|isn't|wasn't|aren't|wouldn't|"
    r"couldn't|can't|cannot|doesn't|does|did|didn't|seems?|sounds?|feels?|looks?|needs?|requires?|costs?|risks?|"
    r"means?|makes?|gives?|pays?|gets?|leaves?|loses?|exposes?|helps?|hurts?|works?|has|have|had|shows?|offers?|"
    r"brings?|yields?|earns?|carries|involves?|gains?|wins?|beats?|tempts?|matters?|adds?|comes?|goes|keeps?|"
    r"leads?|guarantees?|tends?)\b"
)
_MODAL_NEXT = re.compile(
    r"^\s*(?:would|could|might|may|can't|cannot|couldn't|wouldn't|shouldn't|should|won't|isn't|is not|aren't|"
    r"doesn't|does not|needs?|requires?|is too|seems too|is risky|is dangerous|is pointless|is a bad|is not)\b"
)
_ING_NOUNS = {"nothing", "something", "anything", "everything", "thing", "during", "morning", "evening", "king"}
_DETERMINERS = re.compile(r"^(?:(?:the|a|an|my|this|that|your|our)\s+)*$")


@dataclass
class Mention:
    """One mention of a move word: `status` is COMMITTED or the reason it does not count."""

    move: str
    match: "re.Match[str]"
    status: str

    @property
    def committed(self) -> bool:
        return self.status == COMMITTED

    @property
    def start(self) -> int:
        return self.match.start()


def _clause_start(t: str, pos: int) -> int:
    start = 0
    for rx in (_HARD, _SOFT):
        for m in rx.finditer(t, 0, pos):
            start = max(start, m.end())
    return start


def _clause_end(t: str, pos: int) -> Tuple[int, str]:
    """End of the clause that holds `pos`, and the character that ends it ('' at the end of the text)."""
    ends = [m for rx in (_HARD, _SOFT) for m in [rx.search(t, pos)] if m]
    if not ends:
        return len(t), ""
    first = min(ends, key=lambda m: m.start())
    return first.start(), first.group().strip()


def contexts(t: str, start: int, end: int, stop: Optional[int] = None) -> Tuple[str, str, str]:
    """(left, right, end mark) of the span [start, end) in normalized text `t`. The right context also stops at
    `stop`, where the next move word starts ("I'll place a rose now and bid 2 next turn": "next turn" is the
    bid's); the end mark is the clause's."""
    cs = _clause_start(t, start)
    left = t[cs:start]
    if _AND_BEFORE.search(left):
        left = ""
    ce, mark = _clause_end(t, end)
    if stop is not None and end <= stop < ce:
        ce = stop  # the end mark stays the clause's: "bid 4 5 or liar?" is one question
    return left.strip(), t[end:ce], mark


def _previous_clause(t: str, start: int) -> str:
    """The short clause just before the mention's own, in the same sentence ("Next round, I'll give you 2")."""
    cs = _clause_start(t, start)
    if cs == 0 or not t[:cs].rstrip().endswith(","):
        return ""
    head = t[: cs].rstrip()[:-1]
    prev = head[_clause_start(head, len(head)):]
    return prev if len(prev.split()) <= 4 else ""


def status_of(
    t: str, start: int, end: int, extra: Optional[Pattern[str]] = None, stop: Optional[int] = None
) -> str:
    """Why the mention t[start:end] does not count, or COMMITTED (`stop`: where the next move word starts)."""
    left, right, mark = contexts(t, start, end, stop)
    if _NEGATION.search(left) or _NEGATIVE_OBJECT.search(right):
        return "negated"
    if _PAST_BEFORE.search(left) or _PAST_AFTER.search(right):
        return "past"
    if _LATER.search(left) or _LATER.search(right) or _LATER.search(_previous_clause(t, start)):
        return "later"
    if re.search(r"^\s*then\b", t[_clause_start(t, start):start]) and _clause_start(t, start) > 0:
        if not re.search(r"[.!?;:\n]\s*$", t[: _clause_start(t, start)]):
            return "later"  # "rose, then bid 2": a move after this one
    if mark == "?":
        return "question"
    if _HYPOTHETICAL.search(left) or _CONDITIONAL_AFTER.search(right):
        return "hypothetical"
    words = left.split()
    gerunds = [w.strip("'") for w in (words[:1] + words[-1:] + t[start:end].split()[:1])]
    if any(w.endswith("ing") and w not in _ING_NOUNS for w in gerunds) and _FINITE.search(right):
        return "discussed"
    if _DETERMINERS.match(left) and _MODAL_NEXT.search(right):
        return "discussed"
    if extra is not None and extra.search(left):
        return "reported"
    return COMMITTED


MovePatterns = Dict[str, Union[str, Pattern[str]]]


def mentions(text: str, moves: MovePatterns, extra: Optional[Pattern[str]] = None) -> List[Mention]:
    """Every mention of a move in `text`, in reply order, each with its status. `moves` maps a move name to the
    regular expression of its words, matched against normalize(text); a mention inside an earlier one is
    skipped."""
    t = normalize(text)
    found: List[Tuple[int, int, str, "re.Match[str]"]] = []
    for move, pattern in moves.items():
        rx = re.compile(pattern) if isinstance(pattern, str) else pattern
        for m in rx.finditer(t):
            found.append((m.start(), -m.end(), move, m))
    found.sort(key=lambda x: (x[0], x[1]))
    kept: List[Tuple[int, int, str, "re.Match[str]"]] = []
    covered = -1
    for start, neg_end, move, m in found:
        if start >= covered:
            kept.append((start, -neg_end, move, m))
            covered = -neg_end
    out: List[Mention] = []
    for i, (start, end, move, m) in enumerate(kept):
        stop = kept[i + 1][0] if i + 1 < len(kept) else None
        out.append(Mention(move, m, status_of(t, start, end, extra, stop)))
    return out


def commitment(
    text: str,
    moves: MovePatterns,
    key: Optional[Callable[[Mention], Hashable]] = None,
    extra: Optional[Pattern[str]] = None,
) -> Tuple[Optional[str], Optional[Mention], List[Mention]]:
    """The move a reply commits to.

    Returns (move, its first committed mention, all mentions). `move` is None when the reply commits to no move,
    and 'unclear' when it commits to two different ones; repeating the same move is fine. `key` decides when two
    mentions are the same move (default: the move name; a game with arguments, such as a bid, can compare those
    too).
    """
    found = mentions(text, moves, extra)
    chosen = [m for m in found if m.committed]
    if not chosen:
        return None, None, found
    same = key or (lambda m: m.move)
    if len({same(m) for m in chosen}) > 1:
        return "unclear", None, found
    return chosen[0].move, chosen[0], found
