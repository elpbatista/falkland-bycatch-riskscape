# Code Style Guide

This project follows a clean, minimal Python style focused on readability and consistency.  
The goal is to keep the codebase simple, reproducible, and easy to maintain.

## General Principles

- Prefer simple and explicit code
- Avoid unnecessary abstractions
- Keep modules small and focused
- Write code that is easy to read in one pass

## Formatting

Follow PEP 8 formatting.

Key rules:

- 4 spaces indentation
- Maximum line length ~88 characters
- Use blank lines to separate logical blocks
- Avoid deep nesting when possible

Example:

```python
result = process_data(data)

if result:
    save_output(result)
```

## Imports

Imports should be grouped in this order:

1. Standard library
2. Third-party libraries
3. Project modules

Example:

```python
import datetime
from pathlib import Path

import requests
import geopandas as gpd

from riskscape.config import cfg, paths
```

Avoid wildcard imports.

## Functions

Functions should be small and focused.

Use clear names:

```python
download_file()
load_manifest()
build_layer1()
```

Avoid long functions that perform many tasks.

## Docstrings

Use short PEP-257 style docstrings.

Example:

```python
def daterange(start, end):
    """Yield dates between start and end (inclusive)."""
```

Docstrings should describe what the function does, not restate the code.

## Comments

Use comments sparingly.

Only explain:

- non-obvious logic
- design decisions
- important assumptions

Good example:

```python
# Skip files already listed in the manifest
```

Bad example:

```python
# Increment date by one day
current += datetime.timedelta(days=1)
```

## Section Headers

Avoid decorative comment blocks such as:

```python
# ----------------------------------------------------
```

Prefer simple spacing between functions and logical blocks.

## Configuration

All configurable parameters must come from:

```text
config.yaml
```

Do not hardcode:

- paths
- regions
- datasets
- time ranges

Access configuration through:

```python
from riskscape.config import cfg, paths
```

## Paths

Always resolve paths through the configuration system.

Example:

```python
raw_dir = paths["raw"] / dataset_name
```

Avoid fragile relative paths such as:

```text
../data
```

## Logging and Output

Use simple console output for pipeline progress.

Example:

```python
print("Downloaded:", filename)
```

Avoid excessive logging unless necessary.

## Data Pipeline Structure

The pipeline follows this structure:

```text
config.yaml
      │
      ▼
grid generation
      │
      ▼
data downloads
      │
      ▼
layer construction
      │
      ▼
modeling
```

Each stage should have small, focused scripts.

## Keep the Codebase Lean

Prefer:

- fewer files
- fewer abstractions
- simple modules

Avoid premature complexity.

The code should remain easy to understand by reading it top-to-bottom.
