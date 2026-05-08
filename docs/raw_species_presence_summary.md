# Raw Species Presence Summary

Source file:

```text
data/raw/species_presence/saeri_bbal_safs.csv
```

Processed H3/day/species feature files:

```text
data/features/species_presence/year=<year>/part.parquet
```

Years covered by valid telemetry timestamps: 2022-2023

## Raw Columns

| Column        | Type      | Description             |
|---------------|-----------|-------------------------|
| `BirdID`      | `int64`   | Numeric bird identifier |
| `BirdID_uni`  | `object`  | Unique bird identifier  |
| `TripNum`     | `int64`   | Numeric trip identifier |
| `TripNum_uni` | `object`  | Unique trip identifier  |
| `species`     | `object`  | Species code            |
| `colony`      | `object`  | Colony name             |
| `yearT`       | `int64`   | Telemetry year field    |
| `TagType`     | `object`  | Tag type                |
| `datetime`    | `object`  | Timestamp string        |
| `lat`         | `float64` | Latitude                |
| `lon`         | `float64` | Longitude               |

## Cleaning Summary

| Metric                                         |  Value |
|------------------------------------------------|-------:|
| Raw rows                                       | 59,183 |
| Valid rows after timestamp/coordinate cleaning | 59,182 |
| Invalid datetime rows                          |      1 |
| Rows with missing coordinates                  |      0 |

Timestamps are parsed with:

```text
%m/%d/%y %H:%M
```

and converted to UTC.

## Raw Telemetry By Species

| Species | Records | Individuals | Trips | First timestamp      | Last timestamp       |
|---------|--------:|------------:|------:|----------------------|----------------------|
| BBAL    |  33,425 |          27 |    58 | 2022-12-02 17:37 UTC | 2022-12-17 17:53 UTC |
| SAFS    |  25,757 |          15 |    18 | 2022-10-22 19:24 UTC | 2023-03-16 20:31 UTC |

## Raw Telemetry By Year And Species

| Year | Species | Records | Individuals | Trips | Days |
|-----:|---------|--------:|------------:|------:|-----:|
| 2022 | BBAL    |  33,425 |          27 |    58 |   16 |
| 2022 | SAFS    |  15,295 |          14 |    17 |   71 |
| 2023 | SAFS    |  10,462 |          12 |    12 |   75 |

## Spatial Extent By Species

| Species | Min latitude | Max latitude | Min longitude | Max longitude |
|---------|-------------:|-------------:|--------------:|--------------:|
| BBAL    |   -54.741455 |   -42.808988 |    -68.622485 |    -58.203623 |
| SAFS    |   -54.688900 |   -42.468300 |    -68.136000 |    -31.628900 |

## Individual-Level Distribution

Records, trips, and days are summarized per unique `BirdID_uni`.

| Species | Individuals | Mean records | Median records | 75% records | 90% records | Max records | Mean trips | Median trips | Mean days | Median days |
|---------|------------:|-------------:|---------------:|------------:|------------:|------------:|-----------:|-------------:|----------:|------------:|
| BBAL    |          27 |      1,238.0 |        1,145.0 |     1,460.0 |     1,854.0 |     2,196.0 |       2.15 |          2.0 |       7.8 |         7.0 |
| SAFS    |          15 |      1,717.1 |          856.0 |     3,273.0 |     4,043.4 |     4,592.0 |       1.20 |          1.0 |     100.5 |       101.0 |

## Trip-Level Distribution

Records, individuals, and days are summarized per unique `TripNum_uni`.

| Species | Trips | Mean records | Median records | 75% records | 90% records | Max records | Mean individuals | Mean days | Median days |
|---------|------:|-------------:|---------------:|------------:|------------:|------------:|-----------------:|----------:|------------:|
| BBAL    |    58 |        576.3 |          474.0 |       925.3 |     1,160.3 |     1,779.0 |              1.0 |       4.7 |         4.0 |
| SAFS    |    18 |      1,430.9 |          656.5 |     2,334.0 |     4,015.2 |     4,592.0 |              1.0 |      83.9 |        88.5 |

## Processed H3 Species Presence Features

Raw telemetry points are spatially joined to the H3 grid and aggregated by:

```text
h3, date, species
```

Processed columns:

| Column             | Type                  | Description                                             |
|--------------------|-----------------------|---------------------------------------------------------|
| `h3`               | `uint64`              | H3 cell                                                 |
| `date`             | `datetime64[ns, UTC]` | Daily timestamp                                         |
| `species`          | `string`              | Species code                                            |
| `presence_count`   | `uint16`              | Number of telemetry records in the H3/day/species group |
| `individual_count` | `uint16`              | Number of unique birds in the H3/day/species group      |
| `trip_count`       | `uint16`              | Number of unique trips in the H3/day/species group      |

Processed rows:

```text
10,268 h3/date/species rows
```

## Processed Features By Year And Species

| Year | Species | H3/date/species rows | Unique H3 cells | Days | Presence count | Sum individual count | Sum trip count |
|-----:|---------|---------------------:|----------------:|-----:|---------------:|---------------------:|---------------:|
| 2022 | BBAL    |                4,552 |           3,270 |   16 |         21,329 |                4,890 |          4,950 |
| 2022 | SAFS    |                3,083 |           2,381 |   71 |         11,907 |                3,513 |          3,516 |
| 2023 | SAFS    |                2,633 |           1,874 |   75 |          7,588 |                2,635 |          2,635 |

## Processed Features By Species

| Species | H3/date/species rows | Unique H3 cells | Days | Presence count | Sum individual count | Sum trip count |
|---------|---------------------:|----------------:|-----:|---------------:|---------------------:|---------------:|
| BBAL    |                4,552 |           3,270 |   16 |         21,329 |                4,890 |          4,950 |
| SAFS    |                5,716 |           4,024 |  146 |         19,495 |                6,148 |          6,151 |

## Daily Processed Feature Distribution

Daily values are summarized after aggregating each species by date.

| Species | Days | Mean presence count/day | Median presence count/day | Mean H3 cells/day | Median H3 cells/day | Mean individual count/day | Mean trip count/day |
|---------|-----:|------------------------:|--------------------------:|------------------:|--------------------:|--------------------------:|--------------------:|
| BBAL    |   16 |                 1,333.1 |                   1,155.5 |             284.5 |               278.5 |                     305.6 |               309.4 |
| SAFS    |  146 |                   133.5 |                     119.0 |              39.2 |                38.5 |                      42.1 |                42.1 |

## Modeling Notes

The species training table is built from the processed H3 species presence features plus model features:

```text
data/modeling/species_training/year=<year>/part.parquet
```

For each observed species/date combination, the training table includes all H3 cells in the feature grid. Cells without observed telemetry records receive zero support values and zero target value.
