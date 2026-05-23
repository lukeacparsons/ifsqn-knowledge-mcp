# IFSQN Knowledge MCP

Hosted MCP server exposing `search_knowledge_base` for the IFSQN and Elsmar forum knowledge bases.

The service retrieves from one or both Qdrant stores, pulls a larger candidate set than requested, applies lightweight lexical reranking, and returns citation-ready chunks with source metadata.
When FTS database paths are configured, it also retrieves SQLite FTS5/BM25 keyword candidates and merges them with vector results.

## Endpoints

- `GET /health`
- `POST /mcp`

## Required environment

- `OPENAI_API_KEY`
- `IFSQN_QDRANT_URL`
- `ELSMAR_QDRANT_URL`

## Useful optional environment

- `PORT` default `3200`
- `HOST` default `127.0.0.1`
- `MCP_PATH` default `/mcp`
- `MCP_PUBLIC_BASE_URL`
- `IFSQN_COLLECTION` default `ifsqn_forum_posts_large`
- `IFSQN_FTS_DB_PATH`
- `ELSMAR_COLLECTION` default `elsmar_forum_posts_large`
- `ELSMAR_FTS_DB_PATH`
- `OPENAI_EMBEDDING_MODEL` default `text-embedding-3-large`
- `MAX_CANDIDATES` default `60`
- `MAX_RESULTS` default `20`
- `SQLITE3_BIN` default `sqlite3`
