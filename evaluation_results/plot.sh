#!/bin/bash

# List of all your prompt versions
VERSIONS=(
    "baseline"
    "baseline_multi"
    "cot_answer"
    "cot_answer_confidence"
    "cot_answer_confidence_multi"
    "cot_answer_multi"
    "cot_confidence"
    "cot_confidence_multi"
    "scientific"
)

FOLDERS=(
  "llama"
  "src_gemma"
  "src_gemma_27b"
)

MODELNAME=(
  "llama-3.1-8b-instant"
  "gemma-3-4b-it"
  "gemma-3-27b-it"
)

for i in "${!FOLDERS[@]}"; do
    FOLDER=${FOLDERS[$i]}
    MODEL=${MODELNAME[$i]}

    # Ensure the plot output directory exists for this folder
    mkdir -p "plots/$FOLDER"

    for VERSION in "${VERSIONS[@]}"; do
        echo "Processing: $VERSION in $FOLDER using $MODEL"

        # Pass all 3 required arguments to your python script
        python accuracy_vs_confidence.py "$VERSION" "$FOLDER" "$MODEL"

        if [ $? -eq 0 ]; then
            echo "  ✅ Finished $VERSION"
        else
            echo "  ❌ Error in $VERSION"
        fi
    done
done