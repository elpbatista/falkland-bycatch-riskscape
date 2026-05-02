# Bycatch Risk Modeling Approach

## Objective

The goal is to understand and model bycatch risk by separating three independent components:

- Environmental conditions  
- Species presence  
- Fishing activity  

---

## Core Idea

The model is based on a simple principle:

> Species can be present in an area even if there are no vessels.  
> Bycatch only occurs when species presence and fishing activity overlap.

---

## Components

### Environmental Conditions

Environmental variables describe ocean conditions at each location and time.

These include variables such as:

- Sea Surface Temperature (SST)
- Chlorophyll-a (CHL)
- Sea Surface Height (SSH)
- Their gradients and anomalies

These conditions influence both:

- where species are likely to be  
- and the likelihood of bycatch occurring  

---

### Species Presence

This represents the probability of encountering a species under certain environmental conditions.

This is derived from:

- Telemetry (animal tracking data)

This allows estimating:

P(species presence | environmental conditions)

Importantly:

- This is independent of fishing activity  

---

### Bycatch Observations

These are actual recorded bycatch events.

This is obtained from:

- Fisheries observer data  

These observations help identify:

- Under which environmental conditions bycatch occurs  
- Where interactions between species and fisheries have happened  

---

### Fishing Activity

Fishing effort determines where interactions can happen.

This will be derived separately using:

- VMS / AIS data  

---

## Interpretation

The model separates:

- Where species are likely to be (from telemetry + environment)  
- Where fishing occurs (from VMS/AIS)  
- Where bycatch has been observed (from observer data)  

By combining these, the goal is to identify:

> Conditions and locations where species presence and fishing activity overlap, leading to bycatch risk.
