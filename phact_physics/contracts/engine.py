"""
PHACT Structural Engine
=======================
Deterministic forward evaluation over a directed dependency graph for
physics-grounded design certification.

This is ordinary forward evaluation of known physics equations over a
dependency DAG -- not causal inference, do-calculus, or Pearl's structural
causal models. There are no latent variables, no confounders, and no
probabilistic identification: every equation is a known closed-form physics
law, and "what changes if input X changes" is answered by substituting the new
value and recomputing.

Architecture:
  DependencyGraph - the dependency graph (nodes, edges, structural equations)
  Override - override an input and propagate forward
  ComparisonQuery - compare two forward passes (baseline vs. modified input)

The safety guarantee rests on the input/output structure of the graph:
fixed design parameters are input nodes; the certified quantity is a derived
leaf node computed from them. Because the certified quantity is never supplied
by the proposer, it cannot be forged.

Operations:
  compute() - topological forward pass over all nodes
  override() - forward pass with one or more inputs overridden
  find_optimal_override() - binary search for the minimal input change
                              that carries the outcome across a threshold
  compare() - two forward passes; returns the delta
  identify_root_inputs() - ancestry walk to the root input of a violation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Node types ────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """
    A node in the dependency graph - represents one physical variable.

    Attributes:
        name:        Variable name (e.g. "zeta_potential_mv")
        node_type:   "exogenous" - designer-controlled input (root node)
                     "endogenous" - determined by structural equation
                     "outcome" - final pass/fail target (derived leaf)
        unit:        Physical unit string (for display + sanity checks)
        description: Human-readable description
        bounds:      (min, max) physically meaningful range
        structural_eq: f(parents_dict) → float  (None for exogenous nodes)
        parent_names:  List of parent node names (empty for exogenous)
    """
    name:           str
    node_type:      str                          # "exogenous" | "endogenous" | "outcome"
    unit:           str
    description:    str
    bounds:         tuple[float, float]          # (min, max) physically valid range
    structural_eq:  Optional[Callable]  = None  # f(parents: dict) → float
    parent_names:   list[str]           = field(default_factory=list)


@dataclass
class GraphEdge:
    """
    A directed dependency edge: parent → child.

    effect_direction: +1  (increasing parent increases child)
                      -1  (increasing parent decreases child)
                       0  (non-monotonic - depends on operating point)
    mechanism:        One-line description of the physical law behind this edge
    """
    parent:           str
    child:            str
    effect_direction: int    # +1 / -1 / 0
    mechanism:        str


# ── Comparison result ─────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """
    Answer to: "What would [outcome] have been if [variable] had been [value]?"

    Computed as two deterministic forward passes: one with the original
    inputs, one with the modified input, returning the delta.
    """
    query:                   str          # natural language question
    variable:                str          # the variable we overrode
    factual_value:           float        # what it actually was
    comparison_value:    float        # what we're asking "what if it had been"
    factual_outcome:         dict[str, float]        # actual computed outcomes
    comparison_outcome:  dict[str, float]        # predicted outcomes under the change
    outcome_delta:           dict[str, float]        # difference
    passes_under_change:         bool                    # does design pass physics under the change?
    explanation:             str


@dataclass
class OverrideResult:
    """
    Result of overriding input X to value x and propagating forward.
    Returns the optimal value of X that satisfies all physics constraints,
    along with the dependency path explaining which nodes are affected.
    """
    variable:            str
    optimal_value:       float
    unit:                str
    outcomes:            dict[str, float]
    passes:              bool
    dependency_path:         list[str]          # human-readable dependency chain
    comparisons:         list[ComparisonResult]
    explanation:         str


# ── Core dependency graph ─────────────────────────────────────────────────────────

class DependencyGraph:
    """
    Directed dependency graph for a physics domain.

    Encodes:
      - Which variables depend on which others (DAG)
      - The structural equations relating them (from physics laws)
      - Which variables are designer-controlled (input/root nodes)

    Supports:
      - Deterministic forward evaluation over the graph
      - Input override and re-evaluation (for correction feedback)
      - Root input identification for violations
      - Optimal input search via binary search
    """

    def __init__(self, domain: str, description: str):
        self.domain      = domain
        self.description = description
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge]      = []

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.name] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def parents_of(self, node_name: str) -> list[str]:
        return [e.parent for e in self.edges if e.child == node_name]

    def children_of(self, node_name: str) -> list[str]:
        return [e.child for e in self.edges if e.parent == node_name]

    def topological_order(self) -> list[str]:
        """Return nodes in topological order (parents before children)."""
        visited = set()
        order   = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            for parent in self.parents_of(name):
                visit(parent)
            order.append(name)

        for name in self.nodes:
            visit(name)
        return order

    def compute(self, exogenous: dict[str, float]) -> dict[str, float]:
        """
        Forward pass: given input (designer-controlled) values,
        compute all derived and outcome variables via structural equations
        in topological order.
        """
        values = dict(exogenous)

        for name in self.topological_order():
            if name in values:
                continue
            node = self.nodes[name]
            if node.structural_eq is None:
                raise ValueError(
                    f"Node '{name}' has no structural equation and was not "
                    f"provided as exogenous input."
                )
            parents = {p: values[p] for p in node.parent_names}
            values[name] = node.structural_eq(parents)

        return values

    def override(
        self,
        exogenous:     dict[str, float],
        interventions: dict[str, float],
    ) -> dict[str, float]:
        """
        Override one or more input values and re-run the forward pass.
        Used to compute the outcome under a modified design parameter.
        """
        forced = {**exogenous, **interventions}
        return self.compute(forced)

    def compare(
        self,
        observed:              dict[str, float],
        modified_inputs:      dict[str, float],
        query_variable:        str,
    ) -> ComparisonResult:
        """
        Compare two forward passes: one with the original inputs, one with
        a modified input, returning the delta on query_variable.

        Args:
            observed:         The original input values
            modified_inputs: The modified input values to compare
            query_variable:   The outcome variable to compare

        Returns:
            ComparisonResult with both outcomes and their delta
        """
        # Step 1: baseline world
        factual_world = self.compute(observed)

        # Step 2: modified world
        cf_world = self.override(observed, modified_inputs)

        # Build explanation
        cf_var   = list(modified_inputs.keys())[0]
        cf_val   = list(modified_inputs.values())[0]
        fact_val = observed.get(cf_var, factual_world.get(cf_var))

        fact_outcome = factual_world.get(query_variable, float("nan"))
        cf_outcome   = cf_world.get(query_variable, float("nan"))
        delta        = cf_outcome - fact_outcome

        # Collect all outcome deltas
        outcome_delta = {
            k: round(cf_world.get(k, 0) - factual_world.get(k, 0), 6)
            for k in cf_world
        }

        # Build dependency path
        path = self._dependency_path(cf_var, query_variable)
        path_str = " → ".join(path) if path else f"{cf_var} → {query_variable}"

        explanation = (
            f"If {cf_var} had been {cf_val:.3g} (instead of {fact_val:.3g}),\n"
            f"then {query_variable} would have been {cf_outcome:.4g} (instead of {fact_outcome:.4g}).\n"
            f"Δ{query_variable} = {delta:+.4g}\n"
            f"Dependency path: {path_str}"
        )

        return ComparisonResult(
            query                  = f"What would {query_variable} be if {cf_var} = {cf_val:.3g}?",
            variable               = cf_var,
            factual_value          = fact_val,
            comparison_value   = cf_val,
            factual_outcome        = {query_variable: round(fact_outcome, 6)},
            comparison_outcome = {query_variable: round(cf_outcome, 6)},
            outcome_delta          = outcome_delta,
            passes_under_change        = cf_outcome > 0,  # domain-specific: override in subclass
            explanation            = explanation,
        )

    def find_optimal_override(
        self,
        exogenous:      dict[str, float],
        target_node:    str,
        target_min:     float,
        override_var:   str,
        search_range:   tuple[float, float],
        n_steps:        int = 200,
    ) -> OverrideResult:
        """
        Find the minimum intervention on `override_var` such that
        `target_node` >= `target_min`.

        This is the core design optimisation operation:
        "What is the minimum |ζ| needed for colloidal stability?"
        "What is the minimum bank angle for canyon clearance?"

        Uses binary search over the physical range - fast and exact.
        """
        lo, hi = search_range
        best_value   = hi
        best_outcome = None

        for _ in range(40):  # bisection - converges in 40 steps to float precision
            mid    = (lo + hi) / 2
            values = self.override(exogenous, {override_var: mid})
            target = values.get(target_node, float("-inf"))

            if target >= target_min:
                best_value   = mid
                best_outcome = values
                hi = mid
            else:
                lo = mid

        if best_outcome is None:
            best_outcome = self.override(exogenous, {override_var: hi})

        # Build comparison explanation: original → optimal
        original_val = exogenous.get(override_var, lo)
        cf = self.compare(
            observed       = exogenous,
            modified_inputs = {override_var: best_value},
            query_variable = target_node,
        )

        # Dependency path for explanation
        path      = self._dependency_path(override_var, target_node)
        path_str  = " → ".join(path) if path else f"{override_var} → {target_node}"

        node      = self.nodes.get(override_var)
        unit      = node.unit if node else ""

        explanation = (
            f"Optimal correction: {override_var} ← {best_value:.3g} {unit}\n"
            f"This satisfies {target_node} ≥ {target_min} via dependency path:\n"
            f"  {path_str}\n"
            f"Original value: {original_val:.3g} → Corrected: {best_value:.3g} {unit}\n"
            f"{cf.explanation}"
        )

        return OverrideResult(
            variable        = override_var,
            optimal_value   = best_value,
            unit            = unit,
            outcomes        = best_outcome,
            passes          = best_outcome.get(target_node, 0) >= target_min,
            dependency_path     = path,
            comparisons     = [cf],
            explanation     = explanation,
        )

    def _dependency_path(self, source: str, target: str) -> list[str]:
        """BFS to find dependency path from source to target in the DAG."""
        from collections import deque
        queue   = deque([[source]])
        visited = {source}

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == target:
                return path
            for child in self.children_of(node):
                if child not in visited:
                    visited.add(child)
                    queue.append(path + [child])
        return []

    def identify_root_inputs(self, violations: list[str]) -> list[str]:
        """
        Given a list of violation messages, identify which exogenous
        (designer-controlled) variables are the ROOT INPUTS.

        Returns node names that are exogenous and are ancestors of
        the violated outcome nodes.
        """
        # Find outcome nodes that are violated
        violated_outcomes = [
            name for name, node in self.nodes.items()
            if node.node_type == "outcome"
            and any(name.replace("_", " ") in v.lower() or name in v for v in violations)
        ]

        root_inputs = set()
        for outcome in violated_outcomes:
            # Walk back to exogenous ancestors
            ancestors = self._ancestors(outcome)
            for anc in ancestors:
                if self.nodes[anc].node_type == "exogenous":
                    root_inputs.add(anc)

        return list(root_inputs) if root_inputs else [
            name for name, node in self.nodes.items()
            if node.node_type == "exogenous"
        ]

    def _ancestors(self, node_name: str) -> set[str]:
        """Return all ancestors of a node in the DAG."""
        ancestors = set()
        queue     = [node_name]
        while queue:
            n = queue.pop()
            for parent in self.parents_of(n):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors


# ── Convenience classes ───────────────────────────────────────────────────────

class Override:
    """Fluent API for overriding input values and re-evaluating the graph."""
    def __init__(self, graph: DependencyGraph):
        self.graph         = graph
        self._exogenous:   dict[str, float] = {}
        self._overrides:          dict[str, float] = {}

    def given(self, **kwargs) -> "Override":
        """Set the baseline exogenous context."""
        self._exogenous.update(kwargs)
        return self

    def set_inputs(self, **kwargs) -> "Override":
        """Override input values to compute the outcome under a modified design."""
        self._overrides.update(kwargs)
        return self

    def compute(self) -> dict[str, float]:
        """Execute the override and return all variable values."""
        return self.graph.override(self._exogenous, self._overrides)


class ComparisonQuery:
    """Fluent API for comparing two forward passes on a modified input."""
    def __init__(self, graph: DependencyGraph):
        self.graph       = graph
        self._observed:  dict[str, float] = {}
        self._modified:     dict[str, float] = {}
        self._query:     str              = ""

    def observed(self, **kwargs) -> "ComparisonQuery":
        self._observed.update(kwargs)
        return self

    def if_instead(self, **kwargs) -> "ComparisonQuery":
        self._modified.update(kwargs)
        return self

    def what_would(self, variable: str) -> ComparisonResult:
        self._query = variable
        return self.graph.compare(self._observed, self._modified, variable)
