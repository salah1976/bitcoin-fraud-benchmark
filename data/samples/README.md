#Sample Dataset

This directory can contain a small sample of the processed benchmark for demonstration purposes.
The sample is not intended to reproduce the paper's reported performance. The sample dataset contains 10,000 transactions
(5,000 fraud and 5,000 non-fraud transactions)
provided exclusively for demonstration and code-validation purposes.

The complete benchmark contains 1,687,072 transactions distributed across six chronological snapshots and will be released separately upon publication.



The full results were obtained on the complete benchmark containing 1,68M transactions across six chronological snapshots.
Suggested sample schema:
```text
tx_hash
snapshot_id
block_height
timestamp
input_count
output_count
input_addr_count
coinbase_flag
has_witness
script_type_encoded
input_addr_concentration
io_count_ratio
tx_weight
avg_input_value
total_input_scaled
log_output_value
fee_ratio
prev_addr_seen_ratio
prev_addr_seen_count
label_final
```
