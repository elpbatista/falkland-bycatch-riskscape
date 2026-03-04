# Dynamic Bycatch Riskscape Framework (Five-Layer System)

This note defines a five-layer modeling architecture for generating dynamic bycatch riskscapes. The system separates (1) physical ocean conditions, (2) species distribution, (3) latent bycatch hazard, (4) species-specific latent risk, and (5) realized operational impact. The framework is designed to operate on a daily spatial grid (e.g., raster or H3) and remain modular so that layers can evolve independently.

---

## Layer 1: Seascapes (Physical Layer)

**Goal:** Represent the daily physical-biogeochemical environment and its spatial structure.

**Layer 1 is locked as:**

- SST -> GHRSST MUR L4  
- Chlorophyll-a -> VIIRS L3 SMI (log10 transformed)  
- SSH (ADT) -> CMEMS ADT  
- Period -> 2014-2023  
- Representation -> Continuous, standardized multivariate state  

**Explicit exclusions (Layer 1):**

- No SLA  
- No nFLH  
- No wind  
- No gradients in this layer  

**Representation (per cell, per day):**

- Ocean state vector: `S(x,t) = [SST, log10(Chl), SSH]`

**Derived (used by downstream layers):**

- Daily gradients: `G(x,t) = grad(S(x,t))`

**Output:**

- Continuous ocean state `S(x,t)` on the daily grid  
- (Optional) Seascape gradients `G(x,t)` computed from `S(x,t)` for use in Layers 2-3  

---

## Layer 2: Species SDM (Ecological Layer)

**Goal:** Predict probability of species presence from ocean structure and tracking data.

**Inputs:**

- Seascape gradients `G(x,t)`  
- Tracking/telemetry data  

**Model:**

- `P_species(x,t) = f(G(x,t), tracking)`

**Meteorology (planned addition):**

Meteorological variables that affect bird movement and distribution (e.g., wind speed and direction) should be included here as additional predictors:

- `P_species(x,t) = f(G(x,t), wind(x,t), tracking)`

**Output:**

- `P_species(x,t)` = probability species is present (or using habitat) at location `x` and time `t`

---

## Layer 3: Latent Hazard (Interaction Layer)

**Goal:** Estimate probability that conditions are bycatch-prone, based on environment and observed bycatch.

**Inputs:**

- Seascape gradients `G(x,t)`  
- Bycatch observations  

**Model:**

- `P_hazard(x,t) = g(G(x,t), bycatch_obs)`

**Output:**

- `P_hazard(x,t)` = probability that conditions at `x,t` are bycatch-prone (latent interaction hazard)

---

## Layer 4: Species Latent Risk

**Goal:** Combine ecological exposure and hazard into species-specific latent risk.

**Definition:**

- `R_species(x,t) = P_hazard(x,t) * P_species(x,t)`

**Output:**

- `R_species(x,t)` = species-specific latent risk surface (independent of fishing effort)

---

## Layer 5: Realized Impact (Operational Layer)

**Goal:** Convert latent risk into realized operational impact using fishing effort.

**Inputs:**

- `P_hazard(x,t)`  
- `P_species(x,t)`  
- Effort `E(x,t)` (AIS/VMS effort surface)  

**Definition:**

- `Impact(x,t) = P_hazard(x,t) * P_species(x,t) * E(x,t)`

**Output:**

- Realized impact surface suitable for dynamic mapping and decision support

---

## System Summary

**Flow:**

1. Environment -> daily seascape state  
2. Seascape gradients + tracking -> `P_species`  
3. Seascape gradients + bycatch observations -> `P_hazard`  
4. `P_hazard * P_species` -> species latent risk  
5. `P_hazard * P_species * Effort` -> realized impact  

**Key equations:**

- `P_species(x,t) = f(G(x,t), wind(x,t), tracking)`  
- `P_hazard(x,t) = g(G(x,t), bycatch_obs)`  
- `R_species(x,t) = P_hazard(x,t) * P_species(x,t)`  
- `Impact(x,t) = P_hazard(x,t) * P_species(x,t) * E(x,t)`
