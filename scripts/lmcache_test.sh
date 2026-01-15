#!/bin/bash
# scripts/lmcache_test.sh
# LMCache integration test script for KV-Bench
set -e

ENDPOINT="${KVBENCH_ENDPOINT:-http://localhost:8000}"
MODEL="${KVBENCH_MODEL:-llama-3.1-8b}"
LMCACHE_URL="${LMCACHE_URL:-lm://localhost:8080}"

echo "══════════════════════════════════════════════"
echo "KV-Bench LMCache Integration Test"
echo "══════════════════════════════════════════════"
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL}"
echo "LMCache URL: ${LMCACHE_URL}"
echo "══════════════════════════════════════════════"

# Check if lmcache_server is available
LMCACHE_AVAILABLE=false
if command -v lmcache_server &> /dev/null; then
    LMCACHE_AVAILABLE=true
    echo "LMCache server found, starting..."
    lmcache_server localhost:8080 &
    LMCACHE_PID=$!
    sleep 3
else
    echo "⚠️  LMCache server not found, using mock connector"
fi

# Start KV-Bench with LMCache connector
echo "Starting KV-Bench server..."
if [ "$LMCACHE_AVAILABLE" = true ]; then
    KVBENCH_CONNECTOR__CONNECTOR_TYPE=lmcache \
    KVBENCH_CONNECTOR__LMCACHE_REMOTE_URL="$LMCACHE_URL" \
    kvbench serve --model "$MODEL" &
else
    KVBENCH_CONNECTOR__CONNECTOR_TYPE=lmcache \
    kvbench serve --model "$MODEL" &
fi
SERVER_PID=$!
sleep 3

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
        [ "$LMCACHE_AVAILABLE" = true ] && kill $LMCACHE_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Run cache hit/miss test
echo ""
echo "Running cache hit/miss test..."
python3 -c "
import httpx
import asyncio
import json

async def test_cache():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # First request - should be cache miss
        print('Request 1 (cache miss expected)...')
        r1 = await client.post('${ENDPOINT}/v1/chat/completions',
            json={
                'model': '${MODEL}',
                'messages': [{'role': 'user', 'content': 'Hello world'}],
                'max_tokens': 10
            })
        print(f'  Status: {r1.status_code}')
        if r1.status_code == 200:
            print('  ✅ Request 1 successful')
        else:
            print(f'  ❌ Request 1 failed: {r1.text}')
            return False

        # Same prefix - should be cache hit
        print('Request 2 (cache hit expected - same prefix)...')
        r2 = await client.post('${ENDPOINT}/v1/chat/completions',
            json={
                'model': '${MODEL}',
                'messages': [{'role': 'user', 'content': 'Hello world, how are you?'}],
                'max_tokens': 10
            })
        print(f'  Status: {r2.status_code}')
        if r2.status_code == 200:
            print('  ✅ Request 2 successful')
        else:
            print(f'  ❌ Request 2 failed: {r2.text}')
            return False

        # Different prefix - should be cache miss
        print('Request 3 (cache miss expected - different prefix)...')
        r3 = await client.post('${ENDPOINT}/v1/chat/completions',
            json={
                'model': '${MODEL}',
                'messages': [{'role': 'user', 'content': 'Goodbye world'}],
                'max_tokens': 10
            })
        print(f'  Status: {r3.status_code}')
        if r3.status_code == 200:
            print('  ✅ Request 3 successful')
        else:
            print(f'  ❌ Request 3 failed: {r3.text}')
            return False

        # Check metrics
        print('Checking metrics...')
        metrics = await client.get('${ENDPOINT}/metrics')
        if metrics.status_code == 200:
            data = metrics.json()
            print(f'  Total requests: {data.get(\"requests_total\", \"N/A\")}')
            print(f'  Cache hits: {data.get(\"cache_hits\", \"N/A\")}')
            print(f'  Cache misses: {data.get(\"cache_misses\", \"N/A\")}')

        return True

success = asyncio.run(test_cache())
exit(0 if success else 1)
"

TEST_RESULT=$?

# Cleanup
echo ""
echo "Stopping servers..."
kill $SERVER_PID 2>/dev/null || true
[ "$LMCACHE_AVAILABLE" = true ] && kill $LMCACHE_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "✅ LMCache test complete"
    exit 0
else
    echo ""
    echo "❌ LMCache test failed"
    exit 1
fi
