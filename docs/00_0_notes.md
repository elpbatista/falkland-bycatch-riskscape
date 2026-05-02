# Notes

The current system estimates a relative bycatch risk index derived from
the spatial and temporal co-occurrence of species use and fishing activity.
This index is not a direct estimate of bycatch probability, as it does not
incorporate observed bycatch events.

Future work will integrate observed bycatch data to calibrate the risk index,
enabling the estimation of probabilistic bycatch outcomes and improving the
operational relevance of the system.

```text
predict.py takes the environmental grid,
asks the species model “how much species use here?”,
asks the fishing model “how much fishing activity here?”,
multiplies them,
and saves the resulting risk surface.
```
