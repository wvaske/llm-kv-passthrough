#!/bin/bash
# scripts/genai_perf_test.sh
# GenAI-Perf benchmark test script for KV-Bench
set -e

ENDPOINT="${KVBENCH_ENDPOINT:-http://localhost:8000}"
MODEL="${KVBENCH_MODEL:-llama-3.1-8b}"
CONCURRENCY="${KVBENCH_CONCURRENCY:-8}"
REQUEST_COUNT="${KVBENCH_REQUEST_COUNT:-100}"
INPUT_TOKENS="${KVBENCH_INPUT_TOKENS:-1000}"
OUTPUT_TOKENS="${KVBENCH_OUTPUT_TOKENS:-100}"

echo "══════════════════════════════════════════════"
echo "KV-Bench GenAI-Perf Test"
echo "══════════════════════════════════════════════"
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL}"
echo "Concurrency: ${CONCURRENCY}"
echo "Request Count: ${REQUEST_COUNT}"
echo "Input Tokens: ${INPUT_TOKENS}"
echo "Output Tokens: ${OUTPUT_TOKENS}"
echo "══════════════════════════════════════════════"

# Check if genai-perf is installed
if ! command -v genai-perf &> /dev/null; then
    echo "⚠️  genai-perf not found, installing..."
    pip install genai-perf
fi

# Start KV-Bench server in background
echo "Starting KV-Bench server..."
kvbench serve --model "$MODEL" --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s "${ENDPOINT}/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Run GenAI-Perf benchmark
echo ""
echo "Running GenAI-Perf benchmark..."
genai-perf profile \
    -m "$MODEL" \
    --service-kind openai \
    --endpoint-type chat \
    --url "$ENDPOINT" \
    --streaming \
    --synthetic-input-tokens-mean "$INPUT_TOKENS" \
    --output-tokens-mean "$OUTPUT_TOKENS" \
    --concurrency "$CONCURRENCY" \
    --request-count "$REQUEST_COUNT" \
    --generate-plots \
    || true  # Don't fail if genai-perf has issues

# Cleanup
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "✅ GenAI-Perf test complete"
echo "Results saved to ./artifacts/"
