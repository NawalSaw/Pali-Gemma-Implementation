#!/bin/bash

MODEL_PATH="/path/to/your/model"
PROMPT="Describe this image in detail."
IMAGE_FILE_PATH="/path/to/your/image.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="True"

python inference.py \
    --model_path "$MODEL_PATH" \
    --image_path "$IMAGE_FILE_PATH" \
    --prompt "$PROMPT" \
    --max_tokens $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \