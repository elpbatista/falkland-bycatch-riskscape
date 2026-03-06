# Note: Wind Integration Strategy

## Question

When should wind (and wind regimes) be incorporated into the riskscape framework?

## Decision

Wind will **not** be included in Layer 1 (Physical Seascape State).

Instead, wind will be introduced in **Layer 2 (Derived Physical Forcing Features)**.

## Rationale

Layer 1 represents the **ocean state**:

- SST
- Chl
- SSH

Wind represents **atmospheric forcing**, not ocean state.

Keeping forcing separate from state maintains a clean physical architecture:

Atmosphere (wind) → Ocean response (Layer 1) → Derived interactions (Layer 2) → Biological response → Risk

---

## Bycatch Risk Context

For bycatch risk specifically, wind matters because:

- It influences bird flight energetics  
- It modifies prey aggregation  
- It affects fishing gear deployment  
- It alters vessel behavior  

However, evidence and ecological reasoning suggest that the strongest predictor is often:

**Wind persistence + frontal strength**.

rather than discrete wind regime labels.

This supports using wind as a **continuous forcing variable** combined with ocean gradients, rather than as a categorical regime.

---

## Implementation Plan (Future)

After completing Layer 2 gradients:

1. Add raw wind fields (u, v, speed).
2. Derive wind persistence (3–7 day rolling mean).
3. Derive wind stress magnitude.
4. Combine wind persistence with frontal strength (SST gradient magnitude).
5. Optionally test wind regime classification if needed for interpretation.

---

## Strategic Reminder

Evaluate wind first as:

- A continuous forcing variable
- A persistence signal
- An interaction term with ocean fronts

Classification into wind regimes is optional and primarily useful for communication, not necessarily prediction.

Status: Deferred until Layer 2 completion.
