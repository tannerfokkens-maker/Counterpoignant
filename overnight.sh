#!/bin/bash

uv run bach-gen generate --key "B minor" --model-path models_NEW/finetune_best.pt --mode fugue --style bach --voices 4 --texture polyphonic --imitation high --candidates 200 --temperature 0.9 --min-p 0.03 --max-length 4096

uv run bach-gen generate --key "D minor" --model-path models_NEW/finetune_best.pt --mode fugue --style bach --voices 4 --texture polyphonic --imitation high --candidates 200 --temperature 0.9 --min-p 0.03 --max-length 4096

uv run bach-gen generate --key "Eb major" --model-path models_NEW/finetune_best.pt --mode fugue --style bach --voices 4 --texture polyphonic --imitation high --candidates 200 --temperature 0.9 --min-p 0.03 --max-length 4096

uv run bach-gen generate --key "B minor" --model-path models_NEW/finetune_best.pt --mode chorale --style bach --voices 4 --texture homophonic --imitation none --candidates 200 --temperature 0.9 --min-p 0.03 --max-length 2048

uv run bach-gen generate --key "D minor" --model-path models_NEW/finetune_best.pt --mode invention --style bach --voices 2 --texture polyphonic --imitation high --candidates 200 --temperature 0.9 --min-p 0.03 --max-length 2048

uv run bach-gen generate --key "F# minor" --model-path models_NEW/finetune_best.pt --mode sinfonia --style bach --voices 3 --texture polyphonic --imitation high --candidates 200 --temperature 0.9 --min-p 0.03 --max-length 3072

echo "All done!"
