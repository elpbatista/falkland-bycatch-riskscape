# Rationale for Environmental Variable Selection

## Principle

Variables were selected to represent complementary dimensions of the marine environment that influence species distribution and fishing activity, specifically:

- thermal structure
- biological productivity
- ocean dynamics
- atmospheric forcing

Together, these define a coupled physical–biogeochemical system.

---

## Sea Surface Temperature (SST)

**Represents**:

- Thermal structure of surface waters
- Stratification and water mass identity

**Relevance**:

- Key driver of species habitat ranges
- Influences metabolic rates and prey distribution
- Widely used in marine species distribution models

**Role**:

- Provides the thermal baseline of the system

---

## Chlorophyll-a (CHL)

**Represents**:

- Proxy for phytoplankton biomass
- Indicator of primary productivity

**Relevance**:

- Base of the food web
- Strong predictor of foraging areas for seabirds and marine mammals

**Notes**:

- Typically highly skewed → requires transformation (e.g., log)

**Role**:

- Represents biological resource availability

---

## Sea Surface Height (SSH / ADT)

**Represents**:

- Ocean circulation and mesoscale structure
- Eddies, fronts, and currents

**Relevance**:

- Drives aggregation of prey
- Defines dynamic features important for predator–prey interactions

**Role**:

- Captures physical structure and ocean dynamics

---

## Wind (u10, v10)

**Represents**:

- Atmospheric forcing at the ocean surface

**Relevance**:

For seabirds:

- Controls movement efficiency (e.g., dynamic soaring)
- Shapes foraging range and trajectories

For fishing activity:

- Influences vessel behavior and operational decisions
- Acts as a constraint on accessibility and effort distribution

**Role**:

- Represents behavioral and operational forcing

---

## Integrated Perspective

Individually, the variables represent:

- SST → thermal conditions  
- CHL → biological productivity  
- SSH → physical ocean dynamics  
- Wind → atmospheric forcing  

Together, they describe a coupled system where:

- physical processes structure the environment  
- biological processes respond to those structures  
- movement (animals and vessels) is constrained by forcing  

This supports a data-driven representation of ocean states (seascapes) as emergent combinations of environmental conditions.

---

## Scope and Deliberate Exclusions

The selection focuses on:

- surface-resolved variables
- daily temporal resolution
- globally consistent datasets

The following were not included at this stage:

- salinity
- nutrients (e.g., nitrate, phosphate)
- subsurface variables
- bathymetry

**Rationale**:

- Maintain temporal and spatial consistency (2014–2023)
- Avoid mixing datasets with incompatible resolutions
- Ensure computational tractability
- Preserve a clear, interpretable feature space

---

## Summary Statement

The selected variables (SST, CHL, SSH, wind) provide a coherent and scientifically grounded representation of the marine environment, capturing thermal, biological, physical, and atmospheric processes. This combination supports the identification of dynamic ocean states and provides a suitable foundation for modeling species presence, fishing effort, and bycatch risk.
