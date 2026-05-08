# Raw GFW Fishing Effort Summary

Source files:

```text
data/raw/gfw/year=<year>/fishing_effort.parquet
```

Years covered: 2014-2023

## Raw Columns

| Column        | Type      | Description                                 |
|---------------|-----------|---------------------------------------------|
| `date`        | `int64`   | Timestamp stored as nanoseconds since epoch |
| `hours`       | `float32` | Fishing effort hours                        |
| `lat`         | `float32` | Latitude                                    |
| `lon`         | `float32` | Longitude                                   |
| `flag`        | `object`  | Vessel flag state                           |
| `gear_type`   | `object`  | GFW gear type                               |
| `vessel_id`   | `object`  | Vessel identifier                           |
| `vessel_type` | `object`  | Vessel type                                 |

## Overall Summary

| Metric              |       Value |
|---------------------|------------:|
| Rows                |   2,297,069 |
| Total fishing hours | 3,094,974.5 |
| Unique vessels      |       2,011 |

## Flags By Fishing Hours

| Flag |    Rows |     Hours | Unique vessels |
|------|--------:|----------:|---------------:|
| ARG  | 465,787 | 640,696.7 |            599 |
| CHN  | 343,596 | 532,185.5 |            865 |
| TWN  | 313,881 | 473,311.3 |            233 |
| KOR  | 330,799 | 455,389.6 |             87 |
| ESP  | 370,705 | 444,354.8 |             48 |
| FLK  | 375,899 | 415,090.6 |             25 |
| CHL  |  26,019 |  32,817.4 |             12 |
| VUT  |  17,812 |  25,321.7 |             10 |
| UKR  |  17,389 |  22,447.7 |              5 |
| GBR  |  14,099 |  17,812.5 |              3 |

## Gear Types By Fishing Hours

| Gear type          |      Rows |     Hours | Unique vessels |
|--------------------|----------:|----------:|---------------:|
| TRAWLERS           | 1,253,924 | 1,497,210 |            567 |
| SQUID_JIGGER       |   772,660 | 1,256,653 |          1,207 |
| SET_LONGLINES      |   181,518 |   202,871 |             41 |
| FISHING            |    34,746 |    62,657 |            122 |
| CARRIER            |    27,718 |    33,280 |              3 |
| INCONCLUSIVE       |    11,446 |    18,465 |             33 |
| DRIFTING_LONGLINES |     8,092 |    12,650 |             10 |
| NA                 |     2,279 |     5,562 |             17 |
| FIXED_GEAR         |     3,942 |     4,595 |              7 |
| GEAR               |       595 |       792 |              1 |
| POLE_AND_LINE      |        94 |       121 |              2 |
| OTHER              |        55 |       119 |              1 |

## Vessel Types

| Vessel type |      Rows |     Hours | Unique vessels |
|-------------|----------:|----------:|---------------:|
| FISHING     | 2,269,296 | 3,061,575 |          2,007 |
| CARRIER     |    27,718 |    33,280 |              3 |
| OTHER       |        55 |       119 |              1 |

## Yearly Totals

| Year |    Rows | Total hours | Unique vessels | Flags | Gear types | Vessel types |
|-----:|--------:|------------:|---------------:|------:|-----------:|-------------:|
| 2014 | 119,545 |     341,574 |            475 |    16 |          9 |            2 |
| 2015 | 196,614 |     413,949 |            595 |    19 |          9 |            2 |
| 2016 | 153,620 |     219,531 |            515 |    16 |          9 |            2 |
| 2017 | 226,465 |     312,095 |            617 |    18 |          9 |            2 |
| 2018 | 232,510 |     273,409 |            583 |    17 |         10 |            2 |
| 2019 | 256,690 |     302,038 |            611 |    18 |         10 |            2 |
| 2020 | 279,185 |     315,650 |            531 |    18 |         10 |            2 |
| 2021 | 292,400 |     325,813 |            664 |    17 |         10 |            2 |
| 2022 | 278,711 |     310,148 |            627 |    14 |          9 |            2 |
| 2023 | 261,329 |     280,767 |            740 |    17 |          8 |            2 |

## Per-Vessel-Year Fishing Hours

| Year | Vessels |  Mean | Median |   75% |     90% |     95% |     99% |     Max |
|-----:|--------:|------:|-------:|------:|--------:|--------:|--------:|--------:|
| 2014 |     475 | 719.1 |  499.6 | 896.6 | 1,915.1 | 2,649.4 | 3,898.8 | 4,891.1 |
| 2015 |     595 | 695.7 |  461.7 | 940.2 | 1,494.0 | 2,292.8 | 3,509.4 | 5,290.0 |
| 2016 |     515 | 426.3 |  113.2 | 284.8 | 1,394.7 | 2,547.5 | 3,636.6 | 4,966.1 |
| 2017 |     617 | 505.8 |  250.7 | 626.0 | 1,167.7 | 2,310.0 | 3,487.9 | 4,604.0 |
| 2018 |     583 | 469.0 |  188.4 | 717.8 | 1,046.2 | 2,248.6 | 3,341.5 | 4,953.7 |
| 2019 |     611 | 494.3 |  274.9 | 577.0 |   983.8 | 1,978.2 | 3,608.4 | 4,541.7 |
| 2020 |     531 | 594.4 |  275.2 | 850.1 | 1,461.4 | 2,359.6 | 3,665.6 | 4,916.6 |
| 2021 |     664 | 490.7 |  194.0 | 767.9 | 1,304.5 | 2,193.3 | 3,001.9 | 4,311.8 |
| 2022 |     627 | 494.7 |  257.3 | 660.0 | 1,109.3 | 2,109.6 | 3,457.8 | 4,150.1 |
| 2023 |     740 | 379.4 |  120.9 | 472.8 |   970.8 | 1,664.5 | 3,409.9 | 4,275.8 |

## Processed Fishing Tables

The raw GFW records are aggregated to H3/date features in:

```text
data/features/fishing_effort/year=<year>/part.parquet
```

Processed feature columns:

| Column          | Type                  |
|-----------------|-----------------------|
| `h3`            | `uint64`              |
| `date`          | `datetime64[ns, UTC]` |
| `fishing_hours` | `float32`             |
| `vessel_count`  | `uint16`              |

The modeling fishing table adds:

```text
fishing_activity = fishing_hours * vessel_count
```

and is stored in:

```text
data/modeling/fishing_training/year=<year>/part.parquet
```
