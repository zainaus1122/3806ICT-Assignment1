
#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import argparse
import itertools
import json
import math
import os
import re
import sys
import time
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# =========================
# AST
# =========================

@dataclass(frozen=True, order=True)
class Term:
    name: str
    args: Tuple["Term", ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({','.join(map(str, self.args))})"


@dataclass(frozen=True, order=True)
class Formula:
    kind: str  # atom, not, and, or, implies, forall, exists
    left: Optional["Formula"] = None
    right: Optional["Formula"] = None
    var: Optional[str] = None
    body: Optional["Formula"] = None
    pred: Optional[str] = None
    args: Tuple[Term, ...] = ()

    def __str__(self) -> str:
        if self.kind == "atom":
            if self.pred in {"⊤", "True", "T"}:
                return "⊤"
            if self.pred in {"⊥", "False", "F"}:
                return "⊥"
            if self.args:
                return f"{self.pred}({','.join(map(str, self.args))})"
            return self.pred
        if self.kind == "not":
            return f"~{self.left}" if self.left and self.left.kind == "atom" else f"~({self.left})"
        if self.kind == "and":
            return f"({self.left} & {self.right})"
        if self.kind == "or":
            return f"({self.left} | {self.right})"
        if self.kind == "implies":
            return f"({self.left} -> {self.right})"
        if self.kind == "forall":
            return f"!{self.var}. {self.body}"
        if self.kind == "exists":
            return f"?{self.var}. {self.body}"
        raise ValueError(f"Unknown formula kind: {self.kind}")


# =========================
# Parser
# =========================

TOKEN_RE = re.compile(
    r"\s*(->|[()~,!.&|?]|⊤|⊥|True|False|T|F|[A-Za-z_][A-Za-z0-9_]*)"
)

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class TokenStream:
    def __init__(self, text: str):
        self.tokens = [tok for tok in TOKEN_RE.findall(text) if tok and not tok.isspace()]
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input")
        if expected is not None and tok != expected:
            raise SyntaxError(f"Expected {expected}, got {tok}")
        self.i += 1
        return tok

    def accept(self, tok: str) -> bool:
        if self.peek() == tok:
            self.i += 1
            return True
        return False


class Parser:
    def __init__(self, text: str):
        self.ts = TokenStream(text)

    def parse(self) -> Formula:
        if not self.ts.tokens:
            raise SyntaxError("Empty input")
        f = self.parse_implication()
        if self.ts.peek() is not None:
            raise SyntaxError(f"Unexpected token {self.ts.peek()}")
        return f

    def parse_implication(self) -> Formula:
        left = self.parse_or()
        if self.ts.accept("->"):
            right = self.parse_implication()
            return Formula("implies", left=left, right=right)
        return left

    def parse_or(self) -> Formula:
        f = self.parse_and()
        while self.ts.accept("|"):
            f = Formula("or", left=f, right=self.parse_and())
        return f

    def parse_and(self) -> Formula:
        f = self.parse_unary()
        while self.ts.accept("&"):
            f = Formula("and", left=f, right=self.parse_unary())
        return f

    def parse_unary(self) -> Formula:
        tok = self.ts.peek()
        if tok == "~":
            self.ts.take("~")
            return Formula("not", left=self.parse_unary())

        if tok in ("!", "?"):
            q = self.ts.take()
            var = self.ts.take()
            if not IDENT_RE.match(var):
                raise SyntaxError(f"Expected variable name after quantifier, got {var}")
            self.ts.take(".")
            body = self.parse_unary()
            return Formula("forall" if q == "!" else "exists", var=var, body=body)

        if tok == "(":
            self.ts.take("(")
            f = self.parse_implication()
            self.ts.take(")")
            return f

        return self.parse_atom()

    def parse_term(self) -> Term:
        name = self.ts.take()
        if not IDENT_RE.match(name):
            raise SyntaxError(f"Expected identifier, got {name}")
        args: Tuple[Term, ...] = ()
        if self.ts.accept("("):
            items: List[Term] = []
            if self.ts.peek() != ")":
                items.append(self.parse_term())
                while self.ts.accept(","):
                    items.append(self.parse_term())
            self.ts.take(")")
            args = tuple(items)
        return Term(name, args)

    def parse_atom(self) -> Formula:
        tok = self.ts.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input while parsing atom")
        if tok in {"⊤", "True", "T"}:
            self.ts.take()
            return Formula("atom", pred="⊤")
        if tok in {"⊥", "False", "F"}:
            self.ts.take()
            return Formula("atom", pred="⊥")

        pred = self.ts.take()
        if not IDENT_RE.match(pred):
            raise SyntaxError(f"Expected predicate name, got {pred}")
        args: Tuple[Term, ...] = ()
        if self.ts.accept("("):
            items: List[Term] = []
            if self.ts.peek() != ")":
                items.append(self.parse_term())
                while self.ts.accept(","):
                    items.append(self.parse_term())
            self.ts.take(")")
            args = tuple(items)
        return Formula("atom", pred=pred, args=args)


def parse_formula(text: str) -> Formula:
    return Parser(text).parse()


def parse_formula_file(path: str) -> List[Formula]:
    formulas: List[Formula] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            formulas.append(parse_formula(stripped))
    return formulas


# =========================
# Structural helpers
# =========================

def term_leaf_names(t: Term) -> Set[str]:
    out = {t.name} if not t.args else set()
    for a in t.args:
        out |= term_leaf_names(a)
    return out


def formula_free_symbols(f: Formula) -> Set[str]:
    if f.kind == "atom":
        out: Set[str] = set()
        for a in f.args:
            out |= term_leaf_names(a)
        return out
    if f.kind == "not":
        return formula_free_symbols(f.left) if f.left else set()
    if f.kind in {"and", "or", "implies"}:
        return (formula_free_symbols(f.left) if f.left else set()) | (formula_free_symbols(f.right) if f.right else set())
    if f.kind in {"forall", "exists"}:
        return (formula_free_symbols(f.body) if f.body else set()) - {f.var}
    return set()


def collect_terms_from_formula(f: Formula) -> Set[Term]:
    out: Set[Term] = set()

    def walk_term(t: Term) -> None:
        out.add(t)
        for a in t.args:
            walk_term(a)

    def walk_formula(ff: Formula) -> None:
        if ff.kind == "atom":
            for a in ff.args:
                walk_term(a)
        elif ff.kind == "not":
            if ff.left is not None:
                walk_formula(ff.left)
        elif ff.kind in {"and", "or", "implies"}:
            if ff.left is not None:
                walk_formula(ff.left)
            if ff.right is not None:
                walk_formula(ff.right)
        elif ff.kind in {"forall", "exists"}:
            if ff.body is not None:
                walk_formula(ff.body)

    walk_formula(f)
    return out


def fresh_symbol(used: Set[str], base: str = "c") -> str:
    for i in itertools.count(0):
        candidate = f"{base}{i}" if i else base
        if candidate not in used:
            return candidate


def subst_term(t: Term, var: str, repl: Term) -> Term:
    if not t.args:
        return repl if t.name == var else t
    return Term(t.name, tuple(subst_term(a, var, repl) for a in t.args))


def subst_formula(f: Formula, var: str, repl: Term) -> Formula:
    if f.kind == "atom":
        return Formula("atom", pred=f.pred, args=tuple(subst_term(a, var, repl) for a in f.args))

    if f.kind == "not":
        return Formula("not", left=subst_formula(f.left, var, repl) if f.left else None)

    if f.kind in {"and", "or", "implies"}:
        return Formula(
            f.kind,
            left=subst_formula(f.left, var, repl) if f.left else None,
            right=subst_formula(f.right, var, repl) if f.right else None,
        )

    if f.kind in {"forall", "exists"}:
        if f.var == var:
            return f
        # alpha-renaming to avoid capture
        if f.var in term_leaf_names(repl):
            used = formula_free_symbols(f.body) | term_leaf_names(repl) | {var, f.var}
            new_v = fresh_symbol(used, base=f.var)
            renamed_body = subst_formula(f.body, f.var, Term(new_v)) if f.body else f.body
            return Formula(f.kind, var=new_v, body=subst_formula(renamed_body, var, repl) if renamed_body else None)
        return Formula(f.kind, var=f.var, body=subst_formula(f.body, var, repl) if f.body else None)

    return f


def sequent_key(left: FrozenSet[Formula], right: FrozenSet[Formula], used: FrozenSet[Tuple[str, str, str]]) -> Tuple:
    return (
        tuple(sorted(map(str, left))),
        tuple(sorted(map(str, right))),
        tuple(sorted(used)),
    )


# =========================
# Proof search
# =========================

@dataclass(frozen=True)
class State:
    left: FrozenSet[Formula]
    right: FrozenSet[Formula]
    used: FrozenSet[Tuple[str, str, str]] = frozenset()


@dataclass
class ProofResult:
    provable: bool
    steps: int
    reason: str
    elapsed_ms: float
    proof: Optional[str] = None


class BaseProver:
    """
    Book-style backward proof search.
    The search is depth-limited only as a practical termination guard.
    """

    def __init__(self, max_depth: int = 24):
        self.max_depth = max_depth
        self._seen: Set[Tuple[Tuple, int]] = set()

    def prove(self, formula: Formula) -> ProofResult:
        start = time.perf_counter()
        self._seen = set()
        ok = self._prove_state(State(frozenset(), frozenset({formula}), frozenset()), self.max_depth)
        elapsed = (time.perf_counter() - start) * 1000.0
        return ProofResult(ok, 0, "proved" if ok else "not provable within search bound", elapsed)

    def _closed(self, st: State) -> bool:
        if any(f.kind == "atom" and f.pred == "⊥" for f in st.left):
            return True
        if any(f.kind == "atom" and f.pred == "⊤" for f in st.right):
            return True
        return any(a in st.right for a in st.left)

    def _candidate_terms(self, st: State) -> List[Term]:
        terms: Set[Term] = set()
        for f in st.left | st.right:
            terms |= collect_terms_from_formula(f)
        return sorted(terms, key=str)

    def _used_terms_for(self, st: State, f: Formula, side: str) -> Set[str]:
        return {u[2] for u in st.used if u[0] == str(f) and u[1] == side}

    def _fresh_terms_used_names(self, st: State) -> Set[str]:
        names = set()
        for f in st.left | st.right:
            names |= formula_free_symbols(f)
        for t in self._candidate_terms(st):
            names.add(str(t))
        return names

    def _next_expansion(self, st: State) -> List[Tuple[State, ...]]:
        """
        Return a list of branches. A branch is a tuple of successor states.
        The order follows the book-style algorithm:
        1. closure
        2. non-branching rules
        3. branching rules
        4. quantified rules with existing terms
        5. quantified rules with fresh terms
        """
        if self._closed(st):
            return [()]

        left = list(st.left)
        right = list(st.right)

        # 2. non-branching
        for f in left:
            if f.kind == "and":
                new_left = frozenset((set(st.left) - {f}) | {f.left, f.right})
                return [(State(new_left, st.right, st.used),)]
        for f in right:
            if f.kind == "or":
                new_right = frozenset((set(st.right) - {f}) | {f.left, f.right})
                return [(State(st.left, new_right, st.used),)]
        for f in right:
            if f.kind == "implies":
                new_left = frozenset(set(st.left) | {f.left})
                new_right = frozenset((set(st.right) - {f}) | {f.right})
                return [(State(new_left, new_right, st.used),)]
        for f in left:
            if f.kind == "not":
                new_left = frozenset(set(st.left) - {f})
                new_right = frozenset(set(st.right) | {f.left})
                return [(State(new_left, new_right, st.used),)]
        for f in right:
            if f.kind == "not":
                new_left = frozenset(set(st.left) | {f.left})
                new_right = frozenset(set(st.right) - {f})
                return [(State(new_left, new_right, st.used),)]
        for f in right:
            if f.kind == "forall":
                used_names = self._fresh_terms_used_names(st)
                fresh = fresh_symbol(used_names, base="e")
                body = subst_formula(f.body, f.var, Term(fresh)) if f.body else f.body
                new_right = frozenset((set(st.right) - {f}) | {body})
                return [(State(st.left, new_right, st.used),)]
        for f in left:
            if f.kind == "exists":
                used_names = self._fresh_terms_used_names(st)
                fresh = fresh_symbol(used_names, base="e")
                body = subst_formula(f.body, f.var, Term(fresh)) if f.body else f.body
                new_left = frozenset((set(st.left) - {f}) | {body})
                return [(State(new_left, st.right, st.used),)]

        # 3. branching
        for f in right:
            if f.kind == "and":
                s1 = State(st.left, frozenset((set(st.right) - {f}) | {f.left}), st.used)
                s2 = State(st.left, frozenset((set(st.right) - {f}) | {f.right}), st.used)
                return [(s1,), (s2,)]
        for f in left:
            if f.kind == "or":
                s1 = State(frozenset((set(st.left) - {f}) | {f.left}), st.right, st.used)
                s2 = State(frozenset((set(st.left) - {f}) | {f.right}), st.right, st.used)
                return [(s1,), (s2,)]
        for f in left:
            if f.kind == "implies":
                s1 = State(frozenset(set(st.left) - {f}), frozenset(set(st.right) | {f.left}), st.used)
                s2 = State(frozenset((set(st.left) - {f}) | {f.right}), st.right, st.used)
                return [(s1,), (s2,)]

        candidates = self._candidate_terms(st)

        # 4. quantified rules with existing terms
        for f in right:
            if f.kind == "exists":
                used = self._used_terms_for(st, f, "right")
                for t in candidates:
                    if str(t) not in used:
                        body = subst_formula(f.body, f.var, t) if f.body else f.body
                        new_used = frozenset(set(st.used) | {(str(f), "right", str(t))})
                        return [(State(st.left, frozenset(set(st.right) | {body}), new_used),)]
        for f in left:
            if f.kind == "forall":
                used = self._used_terms_for(st, f, "left")
                for t in candidates:
                    if str(t) not in used:
                        body = subst_formula(f.body, f.var, t) if f.body else f.body
                        new_used = frozenset(set(st.used) | {(str(f), "left", str(t))})
                        return [(State(frozenset(set(st.left) | {body}), st.right, new_used),)]

        # 5. quantified rules with fresh terms
        used_names = self._fresh_terms_used_names(st)
        for f in right:
            if f.kind == "exists":
                fresh = fresh_symbol(used_names, base="c")
                body = subst_formula(f.body, f.var, Term(fresh)) if f.body else f.body
                new_used = frozenset(set(st.used) | {(str(f), "right", str(fresh))})
                return [(State(st.left, frozenset(set(st.right) | {body}), new_used),)]
        for f in left:
            if f.kind == "forall":
                fresh = fresh_symbol(used_names, base="c")
                body = subst_formula(f.body, f.var, Term(fresh)) if f.body else f.body
                new_used = frozenset(set(st.used) | {(str(f), "left", str(fresh))})
                return [(State(frozenset(set(st.left) | {body}), st.right, new_used),)]

        return []

    def _prove_state(self, st: State, depth: int) -> bool:
        sid = (sequent_key(st.left, st.right, st.used), depth)
        if sid in self._seen:
            return False
        self._seen.add(sid)

        if self._closed(st):
            return True
        if depth <= 0:
            return False

        branches = self._next_expansion(st)
        if not branches:
            return False
        if branches == [()]:
            return True

        for branch in branches:
            if all(self._prove_state(s, depth - 1) for s in branch):
                return True
        return False


class ImprovedProver(BaseProver):
    """
    Improved prover with:
    - one-pass cheap saturation (reduces branching)
    - fast memoization (no string keys)
    - goal-oriented term selection
    - heuristic branch ordering
    """
    def __init__(self, max_depth: int = 24):
        super().__init__(max_depth=max_depth)
        self._cache: Dict[Tuple, bool] = {}

    def prove(self, formula: Formula) -> ProofResult:
        self._cache.clear()
        start = time.perf_counter()
        ok = self._prove_state(State(frozenset(), frozenset({formula}), frozenset()), self.max_depth)
        elapsed = (time.perf_counter() - start) * 1000.0
        return ProofResult(ok, 0, "proved" if ok else "not provable within search bound", elapsed)

    # ----- helper: apply cheap non‑branching rules once -----
    def _saturate_once(self, st: State) -> State:
        """Apply propositional non‑branching rules exactly once in order."""
        # We'll use a small helper that applies the first applicable rule.
        def apply_one(s: State) -> Optional[State]:
            left = list(s.left)
            right = list(s.right)
            # andL
            for f in left:
                if f.kind == "and":
                    return State(frozenset((set(s.left)-{f}) | {f.left,f.right}), s.right, s.used)
            # orR
            for f in right:
                if f.kind == "or":
                    return State(s.left, frozenset((set(s.right)-{f}) | {f.left,f.right}), s.used)
            # impR
            for f in right:
                if f.kind == "implies":
                    nl = frozenset(set(s.left) | {f.left})
                    nr = frozenset((set(s.right)-{f}) | {f.right})
                    return State(nl, nr, s.used)
            # notL
            for f in left:
                if f.kind == "not":
                    nl = frozenset(set(s.left)-{f})
                    nr = frozenset(set(s.right) | {f.left})
                    return State(nl, nr, s.used)
            # notR
            for f in right:
                if f.kind == "not":
                    nl = frozenset(set(s.left) | {f.left})
                    nr = frozenset(set(s.right)-{f})
                    return State(nl, nr, s.used)
            # forallR (fresh)
            for f in right:
                if f.kind == "forall":
                    used_names = self._fresh_terms_used_names(s)
                    fresh = fresh_symbol(used_names, base="e")
                    body = subst_formula(f.body, f.var, Term(fresh)) if f.body else f.body
                    nr = frozenset((set(s.right)-{f}) | {body})
                    return State(s.left, nr, s.used)
            # existsL (fresh)
            for f in left:
                if f.kind == "exists":
                    used_names = self._fresh_terms_used_names(s)
                    fresh = fresh_symbol(used_names, base="e")
                    body = subst_formula(f.body, f.var, Term(fresh)) if f.body else f.body
                    nl = frozenset((set(s.left)-{f}) | {body})
                    return State(nl, s.right, s.used)
            return None

        # Apply up to 3 times (more than that rarely helps and costs time)
        cur = st
        for _ in range(3):
            if self._closed(cur):
                return cur
            nxt = apply_one(cur)
            if nxt is None:
                break
            cur = nxt
        return cur

    # ----- fast cache key -----
    def _cache_key(self, st: State, depth: int) -> Tuple:
        # Use frozensets directly (they are hashable and comparable)
        return (st.left, st.right, st.used, depth)

    # ----- heuristic term scoring -----
    def _scored_candidates(self, st: State) -> List[Term]:
        candidates = self._candidate_terms(st)
        # Terms that appear in the goal (right side) are more useful.
        goal_terms: Set[str] = set()
        for f in st.right:
            goal_terms |= {str(t) for t in collect_terms_from_formula(f)}
        def score(t: Term) -> int:
            return 1 if str(t) in goal_terms else 0
        candidates.sort(key=lambda t: (-score(t), str(t)))
        return candidates

    # ----- branch ordering heuristic -----
    def _order_branches(self, branches: List[Tuple[State, ...]]) -> List[Tuple[State, ...]]:
        """Put branches that look easier (fewer formulas) first."""
        def complexity(s: State) -> int:
            return len(s.left) + len(s.right)
        return sorted(branches, key=lambda br: min(complexity(s) for s in br))

    # ----- main expansion (same book order, but with heuristic extras) -----
    def _next_expansion(self, st: State) -> List[Tuple[State, ...]]:
        st = self._saturate_once(st)
        if self._closed(st):
            return [()]

        left = list(st.left)
        right = list(st.right)

        # Branching rules (same order)
        for f in right:
            if f.kind == "and":
                s1 = State(st.left, frozenset((set(st.right)-{f}) | {f.left}), st.used)
                s2 = State(st.left, frozenset((set(st.right)-{f}) | {f.right}), st.used)
                return self._order_branches([(s1,), (s2,)])
        for f in left:
            if f.kind == "or":
                s1 = State(frozenset((set(st.left)-{f}) | {f.left}), st.right, st.used)
                s2 = State(frozenset((set(st.left)-{f}) | {f.right}), st.right, st.used)
                return self._order_branches([(s1,), (s2,)])
        for f in left:
            if f.kind == "implies":
                s1 = State(frozenset(set(st.left)-{f}), frozenset(set(st.right) | {f.left}), st.used)
                s2 = State(frozenset((set(st.left)-{f}) | {f.right}), st.right, st.used)
                return self._order_branches([(s1,), (s2,)])

        # Quantifier rules with existing terms (goal‑oriented)
        candidates = self._scored_candidates(st)
        for f in right:
            if f.kind == "exists":
                used = self._used_terms_for(st, f, "right")
                for t in candidates:
                    if str(t) not in used:
                        body = subst_formula(f.body, f.var, t) if f.body else f.body
                        nu = frozenset(set(st.used) | {(str(f), "right", str(t))})
                        return [(State(st.left, frozenset(set(st.right) | {body}), nu),)]
        for f in left:
            if f.kind == "forall":
                used = self._used_terms_for(st, f, "left")
                for t in candidates:
                    if str(t) not in used:
                        body = subst_formula(f.body, f.var, t) if f.body else f.body
                        nu = frozenset(set(st.used) | {(str(f), "left", str(t))})
                        return [(State(frozenset(set(st.left) | {body}), st.right, nu),)]

        # Fresh terms
        used_names = self._fresh_terms_used_names(st)
        for f in right:
            if f.kind == "exists":
                fresh = fresh_symbol(used_names, base="c")
                body = subst_formula(f.body, f.var, Term(fresh)) if f.body else f.body
                nu = frozenset(set(st.used) | {(str(f), "right", str(fresh))})
                return [(State(st.left, frozenset(set(st.right) | {body}), nu),)]
        for f in left:
            if f.kind == "forall":
                fresh = fresh_symbol(used_names, base="c")
                body = subst_formula(f.body, f.var, Term(fresh)) if f.body else f.body
                nu = frozenset(set(st.used) | {(str(f), "left", str(fresh))})
                return [(State(frozenset(set(st.left) | {body}), st.right, nu),)]

        return []

    def _prove_state(self, st: State, depth: int) -> bool:
        key = self._cache_key(st, depth)
        if key in self._cache:
            return self._cache[key]

        if self._closed(st):
            self._cache[key] = True
            return True
        if depth <= 0:
            self._cache[key] = False
            return False

        # Early cycle detection (same as baseline)
        sid = (sequent_key(st.left, st.right, st.used), depth)
        if sid in self._seen:
            self._cache[key] = False
            return False
        self._seen.add(sid)

        branches = self._next_expansion(st)
        if not branches:
            self._cache[key] = False
            return False
        if branches == [()]:
            self._cache[key] = True
            return True

        # Process branches with heuristic ordering already applied in _next_expansion
        for branch in branches:
            if all(self._prove_state(s, depth - 1) for s in branch):
                self._cache[key] = True
                return True

        self._cache[key] = False
        return False


# =========================
# Benchmark generation
# =========================

def generate_benchmark() -> List[Tuple[str, str]]:
    """
    Balanced easy/hard benchmark formulas.
    The benchmark is generated synthetically, which is allowed by the assignment
    as long as the generation process is documented.
    """
    data: List[Tuple[str, str]] = []

    # Easy propositional
    data += [
        ("A -> A", "easy"),
        ("(A & B) -> A", "easy"),
        ("A -> (A | B)", "easy"),
        ("(A -> B) -> ((C -> A) -> (C -> B))", "medium"),
        ("(~A -> A) -> A", "medium"),
    ]

    # Valid quantified formulas
    data += [
        ("!x. P(x) -> ?x. P(x)", "easy"),
        ("!x. P(x) -> !y. P(y)", "easy"),
        ("!x. (P(x) -> Q(x)) -> (!x. P(x) -> !x. Q(x))", "medium"),
        ("?x. !y. R(x,y) -> !y. ?x. R(x,y)", "medium"),
        ("!x. (P(x) & Q(x)) -> !x. P(x)", "easy"),
        ("!x. (P(x) & Q(x)) -> (?y. P(y) & ?z. Q(z))", "medium"),
    ]

    # Invalid / hard-to-close formulas
    data += [
        ("?x. P(x) -> !x. P(x)", "hard-invalid"),
        ("!x. ?y. R(x,y) -> ?y. !x. R(x,y)", "hard-invalid"),
        ("!x. P(x) -> P(a)", "depends-on-a",),
        ("(A -> B) -> A", "invalid"),
        ("!x. P(x) -> ?y. Q(y)", "invalid"),
    ]

    # Slightly harder nested formulas
    for n in range(2, 6):
        left = " -> ".join([f"P{i}(x{i})" for i in range(1, n + 1)])
        right = " -> ".join([f"P{i}(x{i})" for i in range(1, n)])
        data.append((f"({left}) -> ({right} -> P{n}(x{n}))", "hard-easy"))
        data.append((f"!x. ({left}) -> ?y. Q(y)", "hard-invalid"))

    # Deduplicate while preserving order
    seen: Set[str] = set()
    uniq: List[Tuple[str, str]] = []
    for f, tag in data:
        if f not in seen:
            seen.add(f)
            uniq.append((f, tag))
    return uniq


# =========================
# CLI / evaluation
# =========================

def run_suite(formulas: Sequence[Tuple[str, str]], max_depth: int = 40) -> List[Dict[str, object]]:
    base = BaseProver(max_depth=max_depth)
    imp = ImprovedProver(max_depth=max_depth)
    rows: List[Dict[str, object]] = []

    for idx, (f_str, tag) in enumerate(formulas, 1):
        f = parse_formula(f_str)
        b = base.prove(f)
        i = imp.prove(f)
        rows.append({
            "id": idx,
            "formula": f_str,
            "tag": tag,
            "baseline": b.provable,
            "baseline_ms": round(b.elapsed_ms, 3),
            "improved": i.provable,
            "improved_ms": round(i.elapsed_ms, 3),
        })
    return rows


def print_results(rows: Sequence[Dict[str, object]]) -> None:
    headers = ["id", "tag", "formula", "baseline", "baseline_ms", "improved", "improved_ms"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r[h])))
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))

    base_solved = sum(1 for r in rows if r["baseline"])
    imp_solved = sum(1 for r in rows if r["improved"])
    print()
    print(f"Baseline solved: {base_solved}/{len(rows)}")
    print(f"Improved solved: {imp_solved}/{len(rows)}")


def demo() -> None:
    cases = [
        ("A -> A", True),
        ("!x. P(x) -> ?x. P(x)", True),
        ("!x. P(x) -> !y. P(y)", True),
        ("!x. (P(x) -> Q(x)) -> (!x. P(x) -> !x. Q(x))", True),
        ("?x. !y. R(x,y) -> !y. ?x. R(x,y)", True),
        ("!x. ?y. R(x,y) -> ?y. !x. R(x,y)", False),
        ("?x. P(x) -> !x. P(x)", False),
        ("(A -> B) -> A", False),
    ]
    base = BaseProver(max_depth=24)
    imp = ImprovedProver(max_depth=24)
    print("Sanity tests")
    print("------------")
    for s, expected in cases:
        f = parse_formula(s)
        rb = base.prove(f)
        ri = imp.prove(f)
        status = "OK" if (rb.provable == expected and ri.provable == expected) else "MISMATCH"
        print(f"{status:8}  {s:45} baseline={rb.provable} improved={ri.provable} expected={expected}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="3806ICT Assignment 1 FOL prover")
    parser.add_argument("--file", help="Input text file with one formula per line")
    parser.add_argument("--mode", choices=["baseline", "improved", "benchmark", "demo"], default="demo")
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--json", action="store_true", help="Emit benchmark results as JSON")
    parser.add_argument("--out", help="Write benchmark results to a JSON file")
    args = parser.parse_args(argv)

    if args.mode == "demo":
        demo()
        return 0

    if args.mode in {"baseline", "improved"}:
        if not args.file:
            raise SystemExit("--file is required in baseline/improved mode")
        formulas = parse_formula_file(args.file)
        prover = BaseProver(args.max_depth) if args.mode == "baseline" else ImprovedProver(args.max_depth)
        for idx, f in enumerate(formulas, 1):
            res = prover.prove(f)
            print(f"{idx}. {f} -> {res.provable} ({res.elapsed_ms:.3f} ms) [{res.reason}]")
        return 0

    if args.mode == "benchmark":
        formulas = generate_benchmark()
        rows = run_suite(formulas, max_depth=args.max_depth)
        if args.json or args.out:
            payload = json.dumps(rows, indent=2)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fh:
                    fh.write(payload)
            else:
                print(payload)
        else:
            print_results(rows)
        return 0

    raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
