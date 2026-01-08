# RAG Indexer

Production-ready indexer for KAHI MongoDB documents to OpenSearch with BGE-M3 embeddings.

## Features

- **Multi-collection support:** Works, persons, affiliations, sources, projects, patents
- **Semantic search:** BGE-M3 embeddings (1024 dimensions)
- **GPU acceleration:** CUDA support for faster embedding generation
- **Parallel processing:** Multi-worker architecture for efficient indexing
- **K-NN search:** HNSW algorithm with OpenSearch K-NN plugin
- **Flexible sampling:** Index custom samples from each collection
- **Built-in commands:** Test connections, index samples, setup indices

## Installation

```bash
cd rag_indexer/
pip install -e .
```

After installation, the `rag-indexer` command will be available.

## Commands

### Test Connections

Test MongoDB and OpenSearch connections:

```bash
rag-indexer test
```

### Index Samples (New!)

Index a custom number of documents from each collection:

```bash
# Index 100 documents from all default collections
rag-indexer sample --limit 100

# Index 500 documents from specific collections
rag-indexer sample --limit 500 --collections works person

# Index 200 from all with CPU only
rag-indexer sample --limit 200 --collections affiliations person works --no-gpu
```

### Setup All Indices

Setup all indices with predefined samples:

```bash
# Default samples
rag-indexer setup

# Custom limit for all
rag-indexer setup --limit 10000

# Without GPU
rag-indexer setup --no-gpu
```

### Index Specific Collection

Index a single collection:

```bash
# Index all works
rag-indexer index --collection works

# Index with GPU and limit
rag-indexer index --collection works --limit 1000 --gpu

# Index without deleting existing index
rag-indexer index --collection person --no-delete

# Custom workers and batch size
rag-indexer index --collection affiliations --workers 30 --batch-size 200
```

## Configuration

Configuration via environment variables or CLI arguments:

```bash
# MongoDB
export MONGODB_URI="mongodb://localhost:27017/"
export MONGODB_DATABASE="kahi"

# OpenSearch
export OPENSEARCH_URL="http://localhost:9200"
export OPENSEARCH_USER="admin"
export OPENSEARCH_PASSWORD="admin"

# Indexing
export INDEX_PREFIX="impactu_marie_agent"
export EMBEDDING_MODEL="BAAI/bge-m3"
export EMBEDDING_DIMENSION="1024"

# GPU
export USE_GPU="true"
export CUDA_VISIBLE_DEVICES="0"

# Processing
export WORKERS="20"
export BATCH_SIZE="100"
```

## Quick Start Examples

### For Development (Small Samples)

```bash
# Test connections first
rag-indexer test

# Index small samples for testing
rag-indexer sample --limit 50
```

### For Production

```bash
# Index specific collection with all data
rag-indexer index --collection affiliations --gpu
rag-indexer index --collection person --gpu --limit 50000
rag-indexer index --collection works --gpu
```

## Collections

| Collection | Content | Typical Size |
|------------|---------|--------------|
| **works** | Academic papers with titles, abstracts, authors, citations | 2M+ |
| **person** | Authors with names, affiliations, research areas | 1.5M+ |
| **affiliations** | Institutions with names, types, locations | 47K+ |
| **sources** | Journals, conferences, publishers | Variable |
| **projects** | Research projects with funding, participants | Variable |
| **patents** | Patent documents with inventors, claims | Variable |

## Performance

- **GPU mode:** ~500-1000 docs/second
- **CPU mode:** ~100-200 docs/second
- **Batch processing:** Reduces memory usage
- **Parallel workers:** Maximizes throughput

## Requirements

- Python 3.11+
- MongoDB 4.4+
- OpenSearch 2.0+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM (16GB+ recommended)
- 4GB+ VRAM for GPU (8GB+ recommended)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | `mongodb://localhost:27017/` | MongoDB connection URI |
| `MONGODB_DATABASE` | `kahi` | MongoDB database name |
| `OPENSEARCH_URL` | `http://localhost:9200` | OpenSearch URL |
| `OPENSEARCH_USER` | - | OpenSearch username |
| `OPENSEARCH_PASSWORD` | - | OpenSearch password |
| `INDEX_PREFIX` | `impactu_marie_agent` | Index name prefix |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model |
| `EMBEDDING_DIMENSION` | `1024` | Embedding vector dimension |
| `USE_GPU` | `true` | Enable GPU acceleration |
| `CUDA_VISIBLE_DEVICES` | `0` | CUDA device(s) to use |
| `WORKERS` | `20` | Number of parallel workers |
| `BATCH_SIZE` | `100` | Batch size for processing |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size and workers:

```bash
rag-indexer index --collection works --batch-size 50 --workers 10
```

### Connection Refused

Check services are running:

```bash
# MongoDB
mongosh --eval "db.adminCommand('ping')"

# OpenSearch
curl http://localhost:9200
```

### Slow Indexing

Enable GPU if available:

```bash
rag-indexer index --collection works --gpu
```

## Colombian Entity Filtering

For internal use by the multi-agent system only. See `COLOMBIAN_INDEXING.md` for details.

## License

MIT
