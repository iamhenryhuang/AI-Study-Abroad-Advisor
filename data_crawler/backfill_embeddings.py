"""Standalone backfill script: fill in the vectors where embedding IS NULL in document_chunks.

The pipeline normally runs with ENABLE_EMBEDDING=off (relying on fts_vector full-text search);
run this when you later want vector retrieval:

    python -m data_crawler.backfill_embeddings [--school-id ucla] [--batch-size 16]

It reuses the backend embedder model: BAAI/bge-m3 (1024 dimensions);
the env var BGE_EMBED_MODEL_PATH can point to a local model path.
"""
import argparse
import os

from .db import get_connection   # also loads backend/.env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--school-id", default=None, help="only fill in the given school (default: all)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="maximum number of rows to process (0 = unlimited)")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    # An empty env var (BGE_EMBED_MODEL_PATH= in .env) counts as unset -> use the default hub id;
    # trust_remote_code=True: needed to load BGE-M3 with newer sentence-transformers.
    model_path = os.getenv("BGE_EMBED_MODEL_PATH") or "BAAI/bge-m3"
    print(f"Loading the model: {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            sql = "SELECT id, chunk_text FROM document_chunks WHERE embedding IS NULL"
            params = []
            if args.school_id:
                sql += " AND school_id = %s"
                params.append(args.school_id)
            sql += " ORDER BY id"
            if args.limit > 0:
                sql += " LIMIT %s"
                params.append(args.limit)
            cur.execute(sql, params)
            rows = cur.fetchall()

        print(f"Embeddings to fill in: {len(rows)} rows")
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            vectors = model.encode([r[1] for r in batch],
                                   batch_size=args.batch_size,
                                   normalize_embeddings=True)
            with conn.cursor() as cur:
                for (chunk_id, _), vec in zip(batch, vectors):
                    cur.execute("UPDATE document_chunks SET embedding = %s WHERE id = %s",
                                (vec.tolist(), chunk_id))
            conn.commit()
            print(f"  Progress {min(i + args.batch_size, len(rows))}/{len(rows)}")
    finally:
        conn.close()
    print("[OK] backfill done")


if __name__ == "__main__":
    main()
