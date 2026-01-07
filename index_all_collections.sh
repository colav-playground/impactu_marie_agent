#!/bin/bash

# Script to index all collections with samples
set -e

LIMIT=1000
WORKERS=20
BATCH_SIZE=10

echo "=========================================="
echo "Starting RAG indexing for all collections"
echo "Limit: $LIMIT documents per collection"
echo "=========================================="
echo ""

# Index works
echo "📚 Indexing works collection..."
impactu_marie_indexer \
    --collection works \
    --limit $LIMIT \
    --gpu \
    --workers $WORKERS \
    --batch-size $BATCH_SIZE

echo ""
echo "✅ Works indexing completed"
echo ""

# Index person
echo "👤 Indexing person collection..."
impactu_marie_indexer \
    --collection person \
    --limit $LIMIT \
    --gpu \
    --workers $WORKERS \
    --batch-size $BATCH_SIZE

echo ""
echo "✅ Person indexing completed"
echo ""

# Index affiliations
echo "🏛️  Indexing affiliations collection..."
impactu_marie_indexer \
    --collection affiliations \
    --limit $LIMIT \
    --gpu \
    --workers $WORKERS \
    --batch-size $BATCH_SIZE

echo ""
echo "✅ Affiliations indexing completed"
echo ""

# Show final statistics
echo "=========================================="
echo "📊 Final Statistics"
echo "=========================================="
curl -s "http://localhost:9200/_cat/indices?v&h=index,docs.count,store.size" | grep impactu

echo ""
echo "🎉 All collections indexed successfully!"
