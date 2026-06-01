#Sample Dataset

This directory can contain a small sample of the processed benchmark for demonstration purposes.
The sample is not intended to reproduce the paper's reported performance. The full results were obtained on the complete benchmark containing 1,687,072 transactions across six chronological snapshots.
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
