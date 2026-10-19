"""End-to-end GraphRAG pipeline on top of pg_graphrag (PostgreSQL 18) and Claude.

The database does the graph work (storage with provenance, entity resolution,
PPR / Leiden / PathRAG, access control, erasure). Claude does the language
work: extraction, community reports and answering with citations.

    pip install -r examples/requirements.txt
    export DATABASE_URL=postgresql:///demo     # ANTHROPIC_API_KEY or `ant auth login`
    python examples/graphrag_pipeline.py ingest  --graph docs ./my_texts/*.txt
    python examples/graphrag_pipeline.py lazy    --graph docs ./my_texts/*.txt   # zero LLM tokens
    python examples/graphrag_pipeline.py reports --graph docs
    python examples/graphrag_pipeline.py ask     --graph docs "質問" [--role analyst]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import anthropic
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

MODEL = "claude-opus-5-5"
# Server-side refusal fallback: a declined request is re-run on Anthropic's
# recommended fallback model inside the same call.
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

EXTRACTION_SYSTEM = """You build a knowledge graph from documents.
Extract the entities and the relationships between them that the text states explicitly.

Rules:
- Use the surface form found in the text as the entity name; do not translate names.
- type: a short upper-case label such as PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, CONCEPT.
- description: one sentence, in the language of the text, saying what *this text* says about the entity or relation.
- Relation type: a short snake_case predicate (e.g. headquartered_in, led_by, acquired).
- valid_from / valid_to: ISO-8601 dates when the text says when a relation started or ended, otherwise "".
- exclusive: true only for relations a subject can have with one object at a time (led_by, located_in, ceo_of).
- weight: 1-10, how strongly the text supports the relation.
- Do not invent facts that the text does not support."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "weight": {"type": "number"},
                    "valid_from": {"type": "string"},
                    "valid_to": {"type": "string"},
                    "exclusive": {"type": "boolean"},
                },
                "required": ["source", "target", "type", "description", "weight", "valid_from", "valid_to", "exclusive"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relations"],
    "additionalProperties": False,
}

REPORT_SYSTEM = """You write community reports for a knowledge graph.
Given the entities and relationships of one community, write:
- title: a short, specific name for the community (in the language of the data)
- summary: 3-6 sentences on what the community is about, its key entities, and notable facts,
  including when facts were valid if dates are given.
Only use information present in the input."""

REPORT_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["title", "summary"],
    "additionalProperties": False,
}

ANSWER_SYSTEM = """Answer the question using only the provided context.
Cite the sources you rely on with their markers exactly as given, e.g. [#12].
If the context does not contain the answer, say so plainly."""


def claude_json(client: anthropic.Anthropic, system: str, user: str, schema: dict, effort: str) -> dict | None:
    """One structured-output call. Returns None when the request was declined."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        # The system prompt is identical for every call: cache it.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        **FALLBACK,
    )
    if response.stop_reason == "refusal":
        print(f"  declined: {response.stop_details}", file=sys.stderr)
        return None
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def add_documents(conn: psycopg.Connection, graph: str, paths: list[str], acl: list[str] | None) -> list[int]:
    doc_ids = []
    for p in paths:
        path = pathlib.Path(p)
        doc_id = conn.execute(
            "SELECT graphrag.add_document(%s, content => %s, external_id => %s, title => %s, acl => %s)",
            (graph, path.read_text(encoding="utf-8"), str(path), path.name, acl),
        ).fetchone()[0]
        doc_ids.append(doc_id)
    return doc_ids


def cmd_ingest(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("SELECT graphrag.create_graph(%s)", (args.graph,))
        doc_ids = add_documents(conn, args.graph, args.files, args.acl)
        chunks = conn.execute(
            "SELECT id, content FROM graphrag.chunk WHERE document_id = ANY(%s) ORDER BY id", (doc_ids,)
        ).fetchall()
        for chunk_id, content in chunks:
            extraction = claude_json(client, EXTRACTION_SYSTEM, content, EXTRACTION_SCHEMA, effort="medium")
            if extraction is None:
                continue
            result = conn.execute("SELECT graphrag.ingest_extraction(%s, %s)", (chunk_id, Jsonb(extraction))).fetchone()[0]
            print(f"chunk {chunk_id}: {result}")
            conn.commit()  # each chunk is durable on its own
        # Candidate duplicates that canonical keys could not catch (review, then merge).
        for row in conn.execute("SELECT entity_a, name_a, entity_b, name_b, round(score::numeric, 3) FROM graphrag.find_duplicates(%s)", (args.graph,)):
            print("possible duplicate:", row)


def cmd_lazy(args: argparse.Namespace) -> None:
    """LLM-free indexing: keyphrase co-occurrence graph, built inside the database."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("SELECT graphrag.create_graph(%s)", (args.graph,))
        doc_ids = add_documents(conn, args.graph, args.files, args.acl)
        for doc_id in doc_ids:
            print(conn.execute("SELECT graphrag.lazy_index(%s, %s)", (args.graph, doc_id)).fetchone()[0])


def cmd_reports(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        print(conn.execute("SELECT graphrag.build_communities(%s)", (args.graph,)).fetchone()[0])
        conn.commit()
        # Only new or changed communities need a (paid) LLM call.
        pending = conn.execute(
            "SELECT p.ref_id FROM graphrag.pending_report p JOIN graphrag.graph g ON g.id = p.graph_id "
            "WHERE g.name = %s AND p.kind = 'community' ORDER BY p.level DESC, p.rank DESC",
            (args.graph,),
        ).fetchall()
        for (community_id,) in pending:
            context = conn.execute("SELECT graphrag.community_context(%s)", (community_id,)).fetchone()[0]
            report = claude_json(client, REPORT_SYSTEM, context, REPORT_SCHEMA, effort="medium")
            if report is None:
                continue
            conn.execute(
                "SELECT graphrag.set_report('community', %s, %s, title => %s, generated_by => %s)",
                (community_id, report["summary"], report["title"], MODEL),
            )
            conn.commit()
            print(f"community {community_id}: {report['title']}")


def cmd_ask(args: argparse.Namespace) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        if args.role:
            # Retrieval runs with the end user's privileges: evidence they may
            # not read never reaches the prompt.
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(args.role)))
        ctx = conn.execute(
            "SELECT graphrag.build_context(%s, query => %s, max_tokens => %s)",
            (args.graph, args.question, args.max_tokens),
        ).fetchone()[0]
    if args.context_only:
        print(ctx["context"])
        return
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": f"<context>\n{ctx['context']}\n</context>\n\nQuestion: {args.question}"}],
        output_config={"effort": "high"},
        **FALLBACK,
    )
    if response.stop_reason == "refusal":
        print(f"declined: {response.stop_details}")
        return
    print("".join(b.text for b in response.content if b.type == "text"))
    print("\nsources:")
    for s in ctx["sections"]["sources"]:
        print(f"  [#{s['chunk_id']}] {s['title'] or s['external_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "lazy"):
        p = sub.add_parser(name)
        p.add_argument("--graph", required=True)
        p.add_argument("--acl", nargs="*", help="roles allowed to read these documents (default: public)")
        p.add_argument("files", nargs="+")
    p = sub.add_parser("reports")
    p.add_argument("--graph", required=True)
    p = sub.add_parser("ask")
    p.add_argument("--graph", required=True)
    p.add_argument("--role", help="PostgreSQL role of the end user")
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--context-only", action="store_true", help="print the retrieved context, do not call the LLM")
    p.add_argument("question")
    args = parser.parse_args()
    {"ingest": cmd_ingest, "lazy": cmd_lazy, "reports": cmd_reports, "ask": cmd_ask}[args.command](args)


if __name__ == "__main__":
    main()
