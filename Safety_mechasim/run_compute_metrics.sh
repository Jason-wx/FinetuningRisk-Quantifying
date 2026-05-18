#!/bin/bash


# Run with default parameters
python compute_metrics.py

# Or run with custom parameters:
# python compute_metrics.py \
#     --base_model_path "/path/to/base_model" \
#     --safer_state_path "/path/to/safer_lora" \
#     --danger_state_path "/path/to/danger_lora" \
#     --checkpoint_dir "./checkpoint" \
#     --checkpoint_start 150 \
#     --checkpoint_end 6150 \
#     --checkpoint_step 150 \
#     --metric_file "./metric.jsonl" \
#     --cuda_device "0"