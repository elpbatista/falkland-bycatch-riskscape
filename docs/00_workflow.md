# Workflow

1. Ingest raw datasets for environmental conditions, species presence, fishing activity, and bycatch events.

2. Clean and filter datasets to remove invalid or missing records.

3. Aggregate all data to the H3 grid at resolution 6.

4. Align all datasets to a common daily temporal resolution.

5. Compute derived variables, including gradients and anomalies.

6. Standardize variables for consistent scaling.

7. Validate the resulting dataset using range, consistency, and statistical checks.

8. Assemble the final dataset at the cell × day level.

9. Apply the modeling framework to estimate hazard and bycatch risk.

10. Generate outputs for analysis and visualization.
