#!/bin/bash
# scripts/coverage_report.sh
# Coverage report script for KV-Bench
set -e

PHASE=${1:-"final"}
MIN_COVERAGE=${2:-85}

echo "══════════════════════════════════════════════"
echo "KV-Bench Coverage Report: Phase ${PHASE}"
echo "══════════════════════════════════════════════"

# Run pytest with coverage
python -m pytest tests/ \
    --cov=src/kvbench \
    --cov-report=term-missing \
    --cov-report=html:coverage_html \
    --cov-report=xml:coverage.xml \
    --cov-branch \
    -v

# Extract coverage percentage
COVERAGE=$(python -m coverage report | grep TOTAL | awk '{print $4}' | tr -d '%')

echo ""
echo "══════════════════════════════════════════════"
echo "Coverage: ${COVERAGE}% (minimum: ${MIN_COVERAGE}%)"
echo "══════════════════════════════════════════════"

# Check if coverage meets minimum
if (( $(echo "$COVERAGE >= $MIN_COVERAGE" | bc -l) )); then
    echo "✅ PASSED - Coverage meets minimum requirement"
    exit 0
else
    echo "❌ FAILED - Coverage below minimum requirement"
    exit 1
fi
