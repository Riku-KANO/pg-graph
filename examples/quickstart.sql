-- pg_graphrag quickstart (PostgreSQL 18)
--   psql -f examples/quickstart.sql
-- Fictional corpus; LLM outputs are given inline as JSON so the script runs
-- without any API key. In production the JSON comes from your extraction prompt.

\set ON_ERROR_STOP on
\pset pager off
DROP EXTENSION IF EXISTS pg_graphrag CASCADE;
CREATE EXTENSION pg_graphrag CASCADE;          -- also installs btree_gist

-- 1. A graph and its documents ------------------------------------------------
SELECT graphrag.create_graph('demo');

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr') THEN CREATE ROLE hr NOLOGIN; END IF;
END $$;

SELECT graphrag.add_document('demo', external_id => 'd1', title => '会社概要',
  content => '株式会社ミナト製作所は横浜市に本社を置く精密機器メーカーである。');
SELECT graphrag.add_document('demo', external_id => 'd2', title => '役員情報',
  content => 'ミナト製作所の代表取締役は山田花子である。山田花子は2020年に就任した。');
SELECT graphrag.add_document('demo', external_id => 'd3', title => '自治体',
  content => '横浜市は神奈川県の県庁所在地であり、港湾都市として知られる。');
SELECT graphrag.add_document('demo', external_id => 'd4', title => '人事記録', acl => ARRAY['hr'],
  content => '人事記録：山田花子は2025年に健康上の理由で休職した。');

-- 2. Store LLM extraction results (one call per chunk) ------------------------
SELECT graphrag.ingest_extraction(c.id, x.extraction::jsonb)
  FROM graphrag.chunk c JOIN graphrag.document d ON d.id = c.document_id
  JOIN (VALUES
    ('d1', '{"entities":[{"name":"株式会社ミナト製作所","type":"organization","description":"横浜市に本社を置く精密機器メーカー"},
                         {"name":"横浜市","type":"location"}],
             "relations":[{"source":"ミナト製作所","target":"横浜市","type":"headquartered in","description":"本社所在地"}]}'),
    ('d2', '{"entities":[{"name":"山田花子","type":"person","description":"ミナト製作所の代表取締役"}],
             "relations":[{"source":"ミナト製作所(株)","target":"山田花子","type":"led_by","description":"代表取締役",
                           "valid_from":"2020-04-01","exclusive":true}]}'),
    ('d3', '{"entities":[{"name":"横浜市","type":"location","description":"神奈川県の県庁所在地"},{"name":"神奈川県","type":"location"}],
             "relations":[{"source":"横浜市","target":"神奈川県","type":"capital_of","description":"県庁所在地"}]}'),
    ('d4', '{"entities":[{"name":"山田花子","type":"person","description":"2025年に休職"},{"name":"休職","type":"status"}],
             "relations":[{"source":"山田花子","target":"休職","type":"status","valid_from":"2025-01-01"}]}')
  ) AS x(ext, extraction) ON x.ext = d.external_id;

-- Canonical keys merged 株式会社ミナト製作所 / ミナト製作所 / ミナト製作所(株):
SELECT name, canonical_key, graphrag.entity_aliases(id) AS aliases FROM graphrag.entity ORDER BY id;
SELECT source, rel_type, target, valid_from, valid_to FROM graphrag.fact ORDER BY id;

-- 3. Multi-hop retrieval (PPR) — the question never mentions 横浜 or 神奈川 ----
SELECT rank, round(score::numeric, 4) AS score, graph_score IS NOT NULL AS via_graph, entities, content
  FROM graphrag.local_search('demo', query => '山田花子が率いる会社の本社がある県は？', top_k => 4);

-- Explainable reasoning path with evidence chunk ids (PathRAG)
SELECT path, round(reliability::numeric, 3) AS reliability, evidence_chunk_ids
  FROM graphrag.relational_paths('demo', ARRAY['山田花子', '神奈川県']);

-- Ready-to-prompt context with citations and a token budget
SELECT graphrag.build_context('demo', query => '山田花子が率いる会社の本社がある県は？', max_tokens => 600)->>'context';

-- 4. Temporal facts: a new CEO closes the previous term (not deleted) --------
SELECT graphrag.upsert_relation('demo', 'ミナト製作所', '鈴木一郎', 'led_by',
                                valid_from => '2026-04-01', exclusive => true, target_type => 'person');
SELECT source, rel_type, target, valid_from, valid_to, is_current FROM graphrag.fact WHERE rel_type = 'led_by';
SELECT path FROM graphrag.relational_paths('demo', ARRAY['ミナト製作所', '山田花子'], as_of => '2023-01-01');

-- 5. Communities (hierarchical Leiden) and LLM reports -----------------------
SELECT graphrag.build_communities('demo');
SELECT kind, ref_id, reason FROM graphrag.pending_report;       -- work queue for the summarizer
-- feed graphrag.community_context(id) to an LLM, then store the result:
SELECT graphrag.set_report('community', c.id, '山田花子の休職を含む、ミナト製作所の経営体制',
                           title => 'ミナト製作所の経営', generated_by => 'example-model')
  FROM graphrag.community c JOIN graphrag.community_member m ON m.community_id = c.id
  JOIN graphrag.entity e ON e.id = m.entity_id WHERE e.name = '休職' LIMIT 1;

-- 6. Permission-aware retrieval -------------------------------------------------
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analyst') THEN CREATE ROLE analyst NOLOGIN; END IF;
END $$;
SELECT graphrag.enable_row_security();
SELECT graphrag.grant_reader('analyst');
SET ROLE analyst;
-- the HR chunk, the 休職 entity and the report built on HR evidence are invisible
SELECT rank, content FROM graphrag.local_search('demo', query => '山田花子 休職', top_k => 5);
SELECT community_id, visibility, summary, members FROM graphrag.global_search('demo', query => '経営');
RESET ROLE;

-- 7. Right to erasure ----------------------------------------------------------
SELECT graphrag.forget_document('demo', 'd4');
SELECT document_uid, report FROM graphrag.erasure_log;
SELECT graphrag.stats('demo');
