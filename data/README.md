#Data

This directory contains benchmark metadata, feature definitions, label statistics, and documentation required to reproduce the experimental protocol.
Files
`feature_definitions.csv`: definitions of the 13 leakage-controlled learning features used in the final models.
`label_statistics.csv`: transaction and fraud-label counts for each chronological snapshot.
`snapshot_ranges.json`: chronological snapshot metadata and temporal split definition.
`sample/`: optional small sample dataset for demonstration purposes.

#Full Dataset Availability
The complete processed benchmark dataset is not stored directly in this GitHub repository because of file-size constraints and redistribution considerations associated with external intelligence sources.


The full benchmark dataset will be deposited separately in a public data repository upon publication. The expected processed release will contain:
`tx_hash`
`snapshot_id`
`block_height`
`timestamp`

the 15 learning features listed in `feature_definitions.csv`
`label_final`
Raw external-intelligence records are not redistributed directly. Fraud labels are derived from temporally valid external intelligence reports available before each transaction timestamp.
#Temporal Protocol

The benchmark follows a strict forward temporal protocol:
Training snapshots: D1, D2, D3
Validation snapshot: D4
Test snapshots: D5, D6
No future transaction, future label, or future external-intelligence information is used during training or validation.

