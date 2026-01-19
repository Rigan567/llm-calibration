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

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting experiments at $(date)"
echo "--------------------------------"

for VERSION in "${VERSIONS[@]}"
do
    echo "Running experiment: $VERSION..."

    # Run the python script and pipe both output and errors to a log file
    # The script uses the updated version of inference_groq_com.py from the previous step
    python inference_openai.py "$VERSION" > "logs/${VERSION}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ Finished $VERSION"
    else
        echo "❌ Error in $VERSION (Check logs/${VERSION}.log)"
    fi
done

echo "--------------------------------"
echo "All experiments completed at $(date)"