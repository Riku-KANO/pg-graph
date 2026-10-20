-- ---------------------------------------------------------------------------
-- pg_graphrag schema (installed after all Rust functions; see lib.rs)
--
-- Design principles
--   * Provenance first: every entity mention and every relation is linked to
--     the chunk(s) it was extracted from. Answers can cite sources, and
--     erasing a document removes exactly the knowledge derived from it.
--   * Evidence-based access control: chunks carry ACLs (PostgreSQL role
--     names). An entity / relation is visible only through evidence the
--     caller may read; LLM-written reports record the ACLs of the evidence
--     they were generated from (required_acls). What an extraction says
--     about an entity (aliases, description, embedding, metadata) is stored
--     on the mention, i.e. under the ACL of its chunk; the entity row only
--     holds its name, type and curated (evidence-free) attributes.
--   * Temporal facts: relations have a valid-time range. PostgreSQL 18's
--     WITHOUT OVERLAPS constraint guarantees that the same fact never has two
--     overlapping versions.
--   * LLM output is data, not truth: reports keep `generated_by`, staleness is
--     tracked automatically so only affected summaries are regenerated.
-- ---------------------------------------------------------------------------

CREATE TABLE graphrag.graph (
    id          integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    config      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE graphrag.document (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    uid          uuid NOT NULL DEFAULT uuidv7() UNIQUE,   -- PostgreSQL 18: time-ordered UUID
    graph_id     integer NOT NULL REFERENCES graphrag.graph(id) ON DELETE CASCADE,
    external_id  text,
    title        text,
    source_uri   text,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    acl          text[],                                  -- NULL / '{}' = public; else role names
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (graph_id, external_id)
);

CREATE TABLE graphrag.chunk (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES graphrag.document(id) ON DELETE CASCADE,
    graph_id     integer NOT NULL REFERENCES graphrag.graph(id) ON DELETE CASCADE,
    ord          integer NOT NULL DEFAULT 0,
    content      text NOT NULL,
    acl          text[],                                  -- inherited from the document unless set
    embedding    real[] CHECK (array_position(embedding, NULL) IS NULL),
    token_count  integer NOT NULL DEFAULT 0,
    lexemes      tsvector,                                -- CJK bigrams + words (graphrag.lexical_tokens)
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, ord)
);
CREATE INDEX chunk_graph_idx ON graphrag.chunk (graph_id);
CREATE INDEX chunk_lexemes_idx ON graphrag.chunk USING gin (lexemes);
CREATE INDEX chunk_restricted_idx ON graphrag.chunk (graph_id) WHERE cardinality(acl) > 0;

CREATE TABLE graphrag.entity (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    graph_id       integer NOT NULL REFERENCES graphrag.graph(id) ON DELETE CASCADE,
    name           text NOT NULL,
    canonical_key  text NOT NULL,                         -- graphrag.canonical_key(name)
    entity_type    text,
    aliases        text[] NOT NULL DEFAULT '{}',          -- curated aliases; extracted ones live on mention
    key_grams      text[] NOT NULL DEFAULT '{}',          -- bigrams of key + aliases (fuzzy lookup)
    embedding      real[] CHECK (array_position(embedding, NULL) IS NULL),  -- curated
    metadata       jsonb NOT NULL DEFAULT '{}'::jsonb,                      -- curated
    is_curated     boolean NOT NULL DEFAULT false,        -- created without chunk evidence: public
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (graph_id, canonical_key)
);
CREATE INDEX entity_key_grams_idx ON graphrag.entity USING gin (key_grams);

-- Entity <-> chunk provenance. `description`, `aliases`, `embedding` and
-- `metadata` are what this chunk says about the entity; they are visible
-- exactly when the chunk is (see graphrag.entity_aliases / entity_embedding).
CREATE TABLE graphrag.mention (
    entity_id    bigint NOT NULL REFERENCES graphrag.entity(id) ON DELETE CASCADE,
    chunk_id     bigint NOT NULL REFERENCES graphrag.chunk(id) ON DELETE CASCADE,
    description  text,
    frequency    integer NOT NULL DEFAULT 1,
    aliases      text[] NOT NULL DEFAULT '{}',            -- surface forms / aliases used in this chunk
    key_grams    text[] NOT NULL DEFAULT '{}',            -- bigrams of the aliases (fuzzy lookup)
    embedding    real[] CHECK (array_position(embedding, NULL) IS NULL),
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX mention_chunk_idx ON graphrag.mention (chunk_id);
CREATE INDEX mention_key_grams_idx ON graphrag.mention USING gin (key_grams);

CREATE TABLE graphrag.relation (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    graph_id        integer NOT NULL REFERENCES graphrag.graph(id) ON DELETE CASCADE,
    source_id       bigint NOT NULL REFERENCES graphrag.entity(id) ON DELETE CASCADE,
    target_id       bigint NOT NULL REFERENCES graphrag.entity(id) ON DELETE CASCADE,
    rel_type        text NOT NULL DEFAULT 'related_to',
    weight          double precision NOT NULL DEFAULT 1.0 CHECK (weight > 0),
    valid_during    tstzrange NOT NULL DEFAULT tstzrange(NULL, NULL),  -- valid time of the fact
    recorded_at     timestamptz NOT NULL DEFAULT now(),                -- transaction time
    invalidated_at  timestamptz,                                       -- when the end was learned
    is_curated      boolean NOT NULL DEFAULT false,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    CHECK (source_id <> target_id),
    CHECK (NOT isempty(valid_during)),
    -- PostgreSQL 18 temporal constraint: one version of a fact per point in time.
    CONSTRAINT relation_fact_no_overlap
        UNIQUE (graph_id, source_id, target_id, rel_type, valid_during WITHOUT OVERLAPS)
);
CREATE INDEX relation_source_idx ON graphrag.relation (source_id);
CREATE INDEX relation_target_idx ON graphrag.relation (target_id);
CREATE INDEX relation_graph_idx ON graphrag.relation (graph_id);

CREATE TABLE graphrag.relation_evidence (
    relation_id  bigint NOT NULL REFERENCES graphrag.relation(id) ON DELETE CASCADE,
    chunk_id     bigint NOT NULL REFERENCES graphrag.chunk(id) ON DELETE CASCADE,
    description  text,
    PRIMARY KEY (relation_id, chunk_id)
);
CREATE INDEX relation_evidence_chunk_idx ON graphrag.relation_evidence (chunk_id);

CREATE TABLE graphrag.community (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    graph_id     integer NOT NULL REFERENCES graphrag.graph(id) ON DELETE CASCADE,
    level        integer NOT NULL,
    parent_id    bigint REFERENCES graphrag.community(id) ON DELETE CASCADE,
    member_hash  text NOT NULL,                           -- identifies the member set across rebuilds
    size         integer NOT NULL,
    rank         double precision NOT NULL DEFAULT 0,     -- structural importance
    built_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX community_graph_level_idx ON graphrag.community (graph_id, level);
CREATE INDEX community_member_hash_idx ON graphrag.community (graph_id, member_hash);

CREATE TABLE graphrag.community_member (
    community_id bigint NOT NULL REFERENCES graphrag.community(id) ON DELETE CASCADE,
    entity_id    bigint NOT NULL REFERENCES graphrag.entity(id) ON DELETE CASCADE,
    PRIMARY KEY (community_id, entity_id)
);
CREATE INDEX community_member_entity_idx ON graphrag.community_member (entity_id);

-- LLM-generated text (entity / relation summaries, community reports).
-- Kept apart from the facts so it can be invalidated, regenerated, erased
-- and access-controlled on its own.
CREATE TABLE graphrag.report (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    graph_id       integer NOT NULL REFERENCES graphrag.graph(id) ON DELETE CASCADE,
    entity_id      bigint UNIQUE REFERENCES graphrag.entity(id) ON DELETE CASCADE,
    relation_id    bigint UNIQUE REFERENCES graphrag.relation(id) ON DELETE CASCADE,
    community_id   bigint UNIQUE REFERENCES graphrag.community(id) ON DELETE CASCADE,
    title          text,
    content        text NOT NULL,
    embedding      real[] CHECK (array_position(embedding, NULL) IS NULL),
    required_acls  jsonb NOT NULL DEFAULT '[]'::jsonb,    -- ACL sets of the evidence used
    stale          boolean NOT NULL DEFAULT false,
    generated_by   text,                                  -- model / pipeline (AI transparency)
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    CHECK (num_nonnulls(entity_id, relation_id, community_id) = 1)
);
CREATE INDEX report_graph_idx ON graphrag.report (graph_id);

-- Which entities each role has been shown (defence against graph-extraction attacks).
CREATE TABLE graphrag.exposure (
    graph_id    integer NOT NULL REFERENCES graphrag.graph(id) ON DELETE CASCADE,
    principal   name NOT NULL,
    entity_id   bigint NOT NULL REFERENCES graphrag.entity(id) ON DELETE CASCADE,
    first_seen  timestamptz NOT NULL DEFAULT now(),
    last_seen   timestamptz NOT NULL DEFAULT now(),
    hits        integer NOT NULL DEFAULT 1,
    PRIMARY KEY (graph_id, principal, entity_id)
);

-- Accountability record for erasure requests (no content is kept).
CREATE TABLE graphrag.erasure_log (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    graph_id      integer,
    document_uid  uuid NOT NULL,
    requested_by  name NOT NULL DEFAULT current_user,
    erased_at     timestamptz NOT NULL DEFAULT now(),
    report        jsonb NOT NULL
);

-- ---------------------------------------------------------------------------
-- Access control helpers
-- ---------------------------------------------------------------------------

-- Is an ACL public (NULL, empty or containing 'public')?
CREATE FUNCTION graphrag.acl_is_public(acl text[]) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT acl IS NULL OR cardinality(acl) = 0 OR 'public' = ANY (acl)
$$;

-- Is a chunk ACL visible to `role`? ACL labels are PostgreSQL role names, so
-- membership (including nested roles) is decided by the database itself.
-- 'USAGE' matches how PostgreSQL itself applies a role's privileges: a
-- membership granted WITH INHERIT FALSE (e.g. the ADMIN-only membership a
-- CREATEROLE user receives in the roles it creates) does not count.
CREATE FUNCTION graphrag.acl_visible_for(role name, acl text[]) RETURNS boolean
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT graphrag.acl_is_public(acl)
        OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles r
             WHERE r.rolname = ANY (acl)
               AND pg_catalog.pg_has_role(role, r.oid, 'USAGE'))
$$;

CREATE FUNCTION graphrag.acl_visible(acl text[]) RETURNS boolean
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT graphrag.acl_visible_for(current_user, acl)
$$;

-- A report is visible when every ACL set of its evidence is satisfied.
CREATE FUNCTION graphrag.acls_satisfied(required jsonb) RETURNS boolean
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT NOT EXISTS (
        SELECT 1 FROM jsonb_array_elements(coalesce(required, '[]'::jsonb)) AS s(acl)
         WHERE NOT graphrag.acl_visible(ARRAY(SELECT jsonb_array_elements_text(s.acl))))
$$;

-- ACL sets of all evidence behind the given entities / relations. SECURITY
-- DEFINER so that the result is complete even when row security hides some
-- of the evidence from the caller (an incomplete answer would under-protect
-- the report).
CREATE FUNCTION graphrag._evidence_acls(p_entity_ids bigint[], p_relation_ids bigint[]) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
    SELECT coalesce(jsonb_agg(DISTINCT to_jsonb(s.acl)), '[]'::jsonb)
      FROM (
        SELECT ARRAY(SELECT x FROM unnest(c.acl) AS x ORDER BY x) AS acl
          FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
         WHERE m.entity_id = ANY (p_entity_ids) AND cardinality(c.acl) > 0
        UNION
        SELECT ARRAY(SELECT x FROM unnest(c.acl) AS x ORDER BY x)
          FROM graphrag.relation_evidence ev JOIN graphrag.chunk c ON c.id = ev.chunk_id
         WHERE ev.relation_id = ANY (p_relation_ids) AND cardinality(c.acl) > 0
      ) s
$$;
REVOKE ALL ON FUNCTION graphrag._evidence_acls(bigint[], bigint[]) FROM PUBLIC;

-- Aliases of an entity that the caller may see: the curated ones plus those
-- recorded with visible evidence. NULL when the entity itself is not visible.
CREATE FUNCTION graphrag.entity_aliases(p_entity_id bigint) RETURNS text[]
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT ARRAY(
        SELECT DISTINCT a FROM (
            SELECT unnest(e.aliases) AS a
            UNION ALL
            SELECT unnest(m.aliases)
              FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
             WHERE m.entity_id = e.id AND graphrag.acl_visible(c.acl)
        ) s ORDER BY a)
      FROM graphrag.entity e
     WHERE e.id = p_entity_id
       AND (e.is_curated OR EXISTS (
            SELECT 1 FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
             WHERE m.entity_id = e.id AND graphrag.acl_visible(c.acl)))
$$;

-- Embedding of an entity as the caller may see it: the curated one, else the
-- most recent one recorded with visible evidence.
CREATE FUNCTION graphrag.entity_embedding(p_entity_id bigint) RETURNS real[]
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT coalesce(
        (SELECT e.embedding FROM graphrag.entity e
          WHERE e.id = p_entity_id
            AND (e.is_curated OR EXISTS (
                 SELECT 1 FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
                  WHERE m.entity_id = e.id AND graphrag.acl_visible(c.acl)))),
        (SELECT m.embedding FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
          WHERE m.entity_id = p_entity_id AND m.embedding IS NOT NULL AND graphrag.acl_visible(c.acl)
          ORDER BY c.id DESC LIMIT 1))
$$;

-- May evidence with ACL `p_acl` end the validity of relation `p_relation_id`
-- (exclusive facts)? Only when everyone who can see the relation can also
-- see the new evidence. Otherwise a restricted document would visibly
-- change a fact that its readers do not share, and leak when it changed.
-- SECURITY DEFINER so that evidence hidden from the ingesting role by row
-- security is taken into account as well.
CREATE FUNCTION graphrag._may_supersede(p_relation_id bigint, p_acl text[]) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
    SELECT graphrag.acl_is_public(p_acl)
        OR NOT EXISTS (
            SELECT 1 FROM graphrag.relation r
             WHERE r.id = p_relation_id
               AND (r.is_curated
                    OR EXISTS (
                        SELECT 1 FROM graphrag.relation_evidence ev JOIN graphrag.chunk c ON c.id = ev.chunk_id
                         WHERE ev.relation_id = r.id
                           AND (graphrag.acl_is_public(c.acl) OR NOT c.acl <@ p_acl))))
$$;
REVOKE ALL ON FUNCTION graphrag._may_supersede(bigint, text[]) FROM PUBLIC;

-- Admit entities into a role's exposure record, honouring a budget expressed
-- as a fraction of the graph. Returns the admitted subset. Ids that are not
-- entities of the graph visible to the role are ignored, so the function
-- cannot be used to probe for hidden entities; the warning carries no counts
-- for the same reason.
CREATE FUNCTION graphrag._exposure_admit(p_graph integer, p_principal name, p_entities bigint[], p_budget double precision)
RETURNS bigint[]
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_known   bigint[];
    v_new     bigint[];
    v_total   bigint;
    v_exposed bigint;
    v_room    bigint;
BEGIN
    IF NOT pg_has_role(session_user, p_principal, 'SET') THEN
        RAISE EXCEPTION 'graphrag: session user % cannot act as %', session_user, p_principal;
    END IF;
    SELECT coalesce(array_agg(entity_id), '{}') INTO v_known
      FROM graphrag.exposure
     WHERE graph_id = p_graph AND principal = p_principal AND entity_id = ANY (p_entities);
    SELECT coalesce(array_agg(e ORDER BY first_ord), '{}') INTO v_new
      FROM (SELECT u.e, min(u.ord) AS first_ord
              FROM unnest(p_entities) WITH ORDINALITY AS u(e, ord)
              JOIN graphrag.entity en ON en.id = u.e AND en.graph_id = p_graph
             WHERE u.e <> ALL (v_known)
               AND (en.is_curated OR EXISTS (
                    SELECT 1 FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id
                     WHERE m.entity_id = en.id AND graphrag.acl_visible_for(p_principal, c.acl)))
             GROUP BY u.e) s;
    IF p_budget > 0 THEN
        SELECT count(*) INTO v_total FROM graphrag.entity WHERE graph_id = p_graph;
        SELECT count(*) INTO v_exposed FROM graphrag.exposure WHERE graph_id = p_graph AND principal = p_principal;
        v_room := greatest(floor(p_budget * v_total)::bigint - v_exposed, 0);
        IF cardinality(v_new) > v_room THEN
            RAISE WARNING 'graphrag: exposure budget reached for role %; results with new entities were withheld',
                p_principal;
            v_new := v_new[1:v_room];
        END IF;
    END IF;
    INSERT INTO graphrag.exposure (graph_id, principal, entity_id)
    SELECT p_graph, p_principal, e FROM unnest(v_new) AS e
    ON CONFLICT DO NOTHING;
    UPDATE graphrag.exposure SET hits = hits + 1, last_seen = now()
     WHERE graph_id = p_graph AND principal = p_principal AND entity_id = ANY (v_known);
    RETURN v_known || v_new;
END
$$;
REVOKE ALL ON FUNCTION graphrag._exposure_admit(integer, name, bigint[], double precision) FROM PUBLIC;

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------

CREATE FUNCTION graphrag._chunk_before_write() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    d record;
BEGIN
    IF TG_OP = 'INSERT' OR NEW.document_id IS DISTINCT FROM OLD.document_id THEN
        SELECT graph_id, acl INTO d FROM graphrag.document WHERE id = NEW.document_id;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'graphrag: document % does not exist', NEW.document_id;
        END IF;
        NEW.graph_id := d.graph_id;
        IF NEW.acl IS NULL THEN
            NEW.acl := d.acl;
        END IF;
    END IF;
    IF TG_OP = 'INSERT' OR NEW.content IS DISTINCT FROM OLD.content THEN
        NEW.lexemes := array_to_tsvector(graphrag.lexical_tokens(NEW.content));
        NEW.token_count := graphrag.estimate_tokens(NEW.content);
    END IF;
    RETURN NEW;
END
$$;
CREATE TRIGGER chunk_before_write BEFORE INSERT OR UPDATE ON graphrag.chunk
    FOR EACH ROW EXECUTE FUNCTION graphrag._chunk_before_write();

-- Chunks that inherited the document ACL follow later changes to it (the
-- chunk trigger below then re-checks the reports). SECURITY DEFINER: the
-- update must reach every inheriting chunk, including ones that row
-- security hides from the role changing the document.
CREATE FUNCTION graphrag._document_acl_sync() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
BEGIN
    UPDATE graphrag.chunk SET acl = NEW.acl
     WHERE document_id = NEW.id AND acl IS NOT DISTINCT FROM OLD.acl;
    RETURN NULL;
END
$$;
CREATE TRIGGER document_acl_sync AFTER UPDATE OF acl ON graphrag.document
    FOR EACH ROW WHEN (OLD.acl IS DISTINCT FROM NEW.acl)
    EXECUTE FUNCTION graphrag._document_acl_sync();

-- Any change of a chunk ACL (inherited or set on the chunk itself) re-checks
-- the reports derived from that chunk. Statement level with transition
-- tables, so re-labelling a whole document refreshes each report once.
CREATE FUNCTION graphrag._chunk_acl_sync() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_chunks bigint[];
BEGIN
    SELECT array_agg(n.id) INTO v_chunks
      FROM new_rows n JOIN old_rows o ON o.id = n.id
     WHERE n.acl IS DISTINCT FROM o.acl;
    IF v_chunks IS NOT NULL THEN
        PERFORM graphrag._refresh_report_acls(
            ARRAY(SELECT DISTINCT m.entity_id FROM graphrag.mention m WHERE m.chunk_id = ANY (v_chunks)),
            ARRAY(SELECT DISTINCT ev.relation_id FROM graphrag.relation_evidence ev WHERE ev.chunk_id = ANY (v_chunks)));
    END IF;
    RETURN NULL;
END
$$;
CREATE TRIGGER chunk_acl_sync AFTER UPDATE ON graphrag.chunk
    REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
    FOR EACH STATEMENT EXECUTE FUNCTION graphrag._chunk_acl_sync();

-- Recompute required_acls of reports that depend on the given entities /
-- relations. SECURITY DEFINER so that reports hidden from the caller by row
-- security are updated too.
CREATE FUNCTION graphrag._refresh_report_acls(p_entity_ids bigint[], p_relation_ids bigint[]) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    r record;
BEGIN
    UPDATE graphrag.report SET required_acls = graphrag._evidence_acls(ARRAY[entity_id], '{}'), stale = true
     WHERE entity_id = ANY (p_entity_ids);
    UPDATE graphrag.report SET required_acls = graphrag._evidence_acls('{}', ARRAY[relation_id]), stale = true
     WHERE relation_id = ANY (p_relation_ids);
    FOR r IN
        SELECT DISTINCT cm.community_id
          FROM graphrag.community_member cm
         WHERE cm.entity_id = ANY (p_entity_ids)
            OR cm.entity_id IN (SELECT rel.source_id FROM graphrag.relation rel WHERE rel.id = ANY (p_relation_ids)
                                UNION
                                SELECT rel.target_id FROM graphrag.relation rel WHERE rel.id = ANY (p_relation_ids))
    LOOP
        UPDATE graphrag.report rep
           SET required_acls = graphrag._evidence_acls(
                   ARRAY(SELECT entity_id FROM graphrag.community_member WHERE community_id = r.community_id),
                   ARRAY(SELECT rel.id FROM graphrag.relation rel
                          WHERE rel.source_id IN (SELECT entity_id FROM graphrag.community_member WHERE community_id = r.community_id)
                            AND rel.target_id IN (SELECT entity_id FROM graphrag.community_member WHERE community_id = r.community_id))),
               stale = true
         WHERE rep.community_id = r.community_id;
    END LOOP;
END
$$;
REVOKE ALL ON FUNCTION graphrag._refresh_report_acls(bigint[], bigint[]) FROM PUBLIC;

CREATE FUNCTION graphrag._entity_before_write() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.canonical_key IS NULL OR NEW.canonical_key = '' THEN
        NEW.canonical_key := graphrag.canonical_key(NEW.name);
    END IF;
    NEW.key_grams := graphrag.entity_grams(NEW.canonical_key, NEW.aliases);
    NEW.updated_at := now();
    RETURN NEW;
END
$$;
CREATE TRIGGER entity_before_write BEFORE INSERT OR UPDATE OF name, canonical_key, aliases ON graphrag.entity
    FOR EACH ROW EXECUTE FUNCTION graphrag._entity_before_write();

CREATE FUNCTION graphrag._mention_before_write() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.key_grams := graphrag.entity_grams(NULL, NEW.aliases);
    RETURN NEW;
END
$$;
CREATE TRIGGER mention_before_write BEFORE INSERT OR UPDATE OF aliases ON graphrag.mention
    FOR EACH ROW EXECUTE FUNCTION graphrag._mention_before_write();

-- Staleness propagation: new or removed evidence makes the summaries of the
-- touched entities, relations and communities stale (statement-level, using
-- transition tables, so bulk loads stay cheap).
CREATE FUNCTION graphrag._stale_from_mentions() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE graphrag.report SET stale = true
     WHERE NOT stale
       AND (entity_id IN (SELECT entity_id FROM changed)
            OR community_id IN (SELECT cm.community_id FROM graphrag.community_member cm
                                 WHERE cm.entity_id IN (SELECT entity_id FROM changed)));
    RETURN NULL;
END
$$;
CREATE TRIGGER mention_stale_ins AFTER INSERT ON graphrag.mention
    REFERENCING NEW TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_mentions();
CREATE TRIGGER mention_stale_upd AFTER UPDATE ON graphrag.mention
    REFERENCING NEW TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_mentions();
CREATE TRIGGER mention_stale_del AFTER DELETE ON graphrag.mention
    REFERENCING OLD TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_mentions();

CREATE FUNCTION graphrag._stale_from_evidence() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE graphrag.report rep SET stale = true
     WHERE NOT rep.stale
       AND (rep.relation_id IN (SELECT relation_id FROM changed)
            OR rep.community_id IN (
                SELECT cm.community_id
                  FROM graphrag.community_member cm
                  JOIN graphrag.relation r ON cm.entity_id IN (r.source_id, r.target_id)
                 WHERE r.id IN (SELECT relation_id FROM changed)));
    RETURN NULL;
END
$$;
CREATE TRIGGER evidence_stale_ins AFTER INSERT ON graphrag.relation_evidence
    REFERENCING NEW TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_evidence();
CREATE TRIGGER evidence_stale_upd AFTER UPDATE ON graphrag.relation_evidence
    REFERENCING NEW TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_evidence();
CREATE TRIGGER evidence_stale_del AFTER DELETE ON graphrag.relation_evidence
    REFERENCING OLD TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_evidence();

-- Structural changes (new / removed / re-timed edges) make community reports stale.
CREATE FUNCTION graphrag._stale_from_relations() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE graphrag.report rep SET stale = true
     WHERE NOT rep.stale
       AND rep.community_id IN (
            SELECT cm.community_id FROM graphrag.community_member cm
             WHERE cm.entity_id IN (SELECT source_id FROM changed UNION SELECT target_id FROM changed));
    RETURN NULL;
END
$$;
CREATE TRIGGER relation_stale_ins AFTER INSERT ON graphrag.relation
    REFERENCING NEW TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_relations();
CREATE TRIGGER relation_stale_upd AFTER UPDATE ON graphrag.relation
    REFERENCING NEW TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_relations();
CREATE TRIGGER relation_stale_del AFTER DELETE ON graphrag.relation
    REFERENCING OLD TABLE AS changed FOR EACH STATEMENT EXECUTE FUNCTION graphrag._stale_from_relations();

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Human-readable facts with validity.
CREATE VIEW graphrag.fact WITH (security_invoker = true) AS
SELECT r.id,
       g.name AS graph,
       s.name AS source,
       r.rel_type,
       t.name AS target,
       r.weight,
       lower(r.valid_during) AS valid_from,
       upper(r.valid_during) AS valid_to,
       r.valid_during @> now() AS is_current,
       r.recorded_at,
       r.invalidated_at,
       (SELECT count(*) FROM graphrag.relation_evidence ev WHERE ev.relation_id = r.id) AS evidence_count
  FROM graphrag.relation r
  JOIN graphrag.graph g ON g.id = r.graph_id
  JOIN graphrag.entity s ON s.id = r.source_id
  JOIN graphrag.entity t ON t.id = r.target_id;

-- Work queue for the (external) LLM summarization pipeline.
CREATE VIEW graphrag.pending_report WITH (security_invoker = true) AS
SELECT c.graph_id, 'community'::text AS kind, c.id AS ref_id, c.level, c.size, c.rank,
       CASE WHEN rep.id IS NULL THEN 'missing' ELSE 'stale' END AS reason
  FROM graphrag.community c
  LEFT JOIN graphrag.report rep ON rep.community_id = c.id
 WHERE rep.id IS NULL OR rep.stale
UNION ALL
SELECT rep.graph_id, CASE WHEN rep.entity_id IS NOT NULL THEN 'entity' ELSE 'relation' END,
       coalesce(rep.entity_id, rep.relation_id), NULL, NULL, NULL, 'stale'
  FROM graphrag.report rep
 WHERE rep.stale AND (rep.entity_id IS NOT NULL OR rep.relation_id IS NOT NULL);

-- ---------------------------------------------------------------------------
-- Erasure
-- ---------------------------------------------------------------------------

-- Erase a document and everything derived from it (see forget_document).
-- Runs with the owner's rights so that the erasure is complete even when row
-- security hides some derived objects (reports built on evidence from several
-- ACLs, entities mentioned only in hidden chunks) from the requesting role.
-- The requester needs the right to delete the document itself: DELETE on
-- graphrag.document and, under row security, an ACL that lets them see it.
CREATE FUNCTION graphrag._erase_document(p_document_id bigint, p_clear_embeddings boolean, p_principal name)
RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
DECLARE
    v_doc              record;
    v_chunks           bigint[];
    v_entities         bigint[];
    v_relations        bigint[];
    v_reports          bigint;
    v_orphan_relations bigint;
    v_orphan_entities  bigint;
    v_embeddings       bigint := 0;
    v_report           jsonb;
BEGIN
    IF NOT pg_has_role(session_user, p_principal, 'SET') THEN
        RAISE EXCEPTION 'graphrag: session user % cannot act as %', session_user, p_principal;
    END IF;
    IF NOT has_table_privilege(p_principal, 'graphrag.document', 'DELETE') THEN
        RAISE EXCEPTION 'graphrag: permission denied to erase documents'
            USING ERRCODE = 'insufficient_privilege';
    END IF;
    SELECT uid, graph_id, acl INTO v_doc FROM graphrag.document WHERE id = p_document_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'graphrag: document % does not exist', p_document_id;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_class WHERE oid = 'graphrag.document'::regclass AND relrowsecurity)
       AND NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = p_principal AND (rolsuper OR rolbypassrls))
       AND NOT graphrag.acl_visible_for(p_principal, v_doc.acl) THEN
        RAISE EXCEPTION 'graphrag: document % does not exist', p_document_id;
    END IF;

    v_chunks := ARRAY(SELECT id FROM graphrag.chunk WHERE document_id = p_document_id);
    v_entities := ARRAY(SELECT DISTINCT entity_id FROM graphrag.mention WHERE chunk_id = ANY (v_chunks));
    v_relations := ARRAY(SELECT DISTINCT relation_id FROM graphrag.relation_evidence WHERE chunk_id = ANY (v_chunks));

    -- 1. LLM text derived from the erased evidence may quote it: delete it.
    DELETE FROM graphrag.report
     WHERE entity_id = ANY (v_entities) OR relation_id = ANY (v_relations)
        OR community_id IN (SELECT community_id FROM graphrag.community_member WHERE entity_id = ANY (v_entities));
    GET DIAGNOSTICS v_reports = ROW_COUNT;
    -- 2. The document; chunks, mentions (with what they said about entities)
    --    and relation evidence cascade.
    DELETE FROM graphrag.document WHERE id = p_document_id;
    -- 3. Facts that no longer have any evidence.
    DELETE FROM graphrag.relation r
     WHERE r.id = ANY (v_relations) AND NOT r.is_curated
       AND NOT EXISTS (SELECT 1 FROM graphrag.relation_evidence ev WHERE ev.relation_id = r.id);
    GET DIAGNOSTICS v_orphan_relations = ROW_COUNT;
    DELETE FROM graphrag.entity e
     WHERE e.id = ANY (v_entities) AND NOT e.is_curated
       AND NOT EXISTS (SELECT 1 FROM graphrag.mention m WHERE m.entity_id = e.id)
       AND NOT EXISTS (SELECT 1 FROM graphrag.relation r WHERE (r.source_id = e.id OR r.target_id = e.id) AND r.is_curated);
    GET DIAGNOSTICS v_orphan_entities = ROW_COUNT;
    -- 4. Entity-level embeddings of surviving entities may encode erased descriptions.
    IF p_clear_embeddings THEN
        UPDATE graphrag.entity SET embedding = NULL
         WHERE id = ANY (v_entities) AND embedding IS NOT NULL;
        GET DIAGNOSTICS v_embeddings = ROW_COUNT;
    END IF;

    v_report := jsonb_build_object(
        'document_uid', v_doc.uid,
        'chunks_deleted', cardinality(v_chunks),
        'entities_touched', cardinality(v_entities),
        'entities_deleted', v_orphan_entities,
        'relations_touched', cardinality(v_relations),
        'relations_deleted', v_orphan_relations,
        'reports_deleted', v_reports,
        'entity_embeddings_cleared', v_embeddings);
    INSERT INTO graphrag.erasure_log (graph_id, document_uid, requested_by, report)
    VALUES (v_doc.graph_id, v_doc.uid, p_principal, v_report);
    RETURN v_report;
END
$$;

-- ---------------------------------------------------------------------------
-- Row-level security and role helpers
-- ---------------------------------------------------------------------------

-- Enforce evidence-based visibility for direct SQL access as well (the
-- retrieval functions always enforce it). Table owners and superusers bypass
-- RLS as usual; application roles do not. An ingestion role that has to
-- write documents for ACLs it does not belong to needs BYPASSRLS (or must
-- own the tables).
CREATE FUNCTION graphrag.enable_row_security() RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    t text;
    p record;
BEGIN
    -- Idempotent: drop this function's previous policies first.
    FOR p IN
        SELECT tablename, policyname FROM pg_catalog.pg_policies
         WHERE schemaname = 'graphrag' AND policyname LIKE 'graphrag\_%'
    LOOP
        EXECUTE format('DROP POLICY %I ON graphrag.%I', p.policyname, p.tablename);
    END LOOP;
    FOREACH t IN ARRAY ARRAY['document', 'chunk', 'mention', 'relation_evidence', 'entity', 'relation', 'report', 'exposure']
    LOOP
        EXECUTE format('ALTER TABLE graphrag.%I ENABLE ROW LEVEL SECURITY', t);
        -- Writes are governed by GRANTs (see grant_writer); reads by evidence.
        EXECUTE format('CREATE POLICY graphrag_insert ON graphrag.%I FOR INSERT WITH CHECK (true)', t);
        EXECUTE format('CREATE POLICY graphrag_update ON graphrag.%I FOR UPDATE USING (true)', t);
        EXECUTE format('CREATE POLICY graphrag_delete ON graphrag.%I FOR DELETE USING (true)', t);
    END LOOP;

    CREATE POLICY graphrag_visible ON graphrag.document FOR SELECT USING (graphrag.acl_visible(acl));
    CREATE POLICY graphrag_visible ON graphrag.chunk FOR SELECT USING (graphrag.acl_visible(acl));
    CREATE POLICY graphrag_visible ON graphrag.mention FOR SELECT
        USING (EXISTS (SELECT 1 FROM graphrag.chunk c WHERE c.id = chunk_id));
    CREATE POLICY graphrag_visible ON graphrag.relation_evidence FOR SELECT
        USING (EXISTS (SELECT 1 FROM graphrag.chunk c WHERE c.id = chunk_id));
    CREATE POLICY graphrag_visible ON graphrag.entity FOR SELECT
        USING (is_curated OR EXISTS (SELECT 1 FROM graphrag.mention m WHERE m.entity_id = id));
    CREATE POLICY graphrag_visible ON graphrag.relation FOR SELECT
        USING ((is_curated OR EXISTS (SELECT 1 FROM graphrag.relation_evidence ev WHERE ev.relation_id = id))
               AND EXISTS (SELECT 1 FROM graphrag.entity e WHERE e.id = source_id)
               AND EXISTS (SELECT 1 FROM graphrag.entity e WHERE e.id = target_id));
    CREATE POLICY graphrag_visible ON graphrag.report FOR SELECT USING (graphrag.acls_satisfied(required_acls));
    CREATE POLICY graphrag_visible ON graphrag.exposure FOR SELECT USING (principal = current_user);
END
$$;

-- Read-only access for an application / end-user role. The retrieval
-- functions run with the caller's rights and therefore need SELECT on the
-- tables; without row security that also lets the role read every chunk
-- directly, so say so.
CREATE FUNCTION graphrag.grant_reader(p_role name) RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    EXECUTE format('GRANT USAGE ON SCHEMA graphrag TO %I', p_role);
    EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA graphrag TO %I', p_role);
    EXECUTE format('GRANT EXECUTE ON FUNCTION graphrag._exposure_admit(integer, name, bigint[], double precision) TO %I', p_role);
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE oid = 'graphrag.chunk'::regclass AND relrowsecurity) THEN
        RAISE WARNING 'graphrag: row level security is off, so role % can read every chunk with a plain SELECT', p_role
            USING HINT = 'Run SELECT graphrag.enable_row_security() unless the role can only reach the graphrag functions.';
    END IF;
END
$$;

-- Read / write access for an ingestion pipeline role (also the maintenance
-- functions that read the whole graph, such as community_context).
CREATE FUNCTION graphrag.grant_writer(p_role name) RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM graphrag.grant_reader(p_role);
    EXECUTE format('GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA graphrag TO %I', p_role);
    EXECUTE format('GRANT USAGE ON ALL SEQUENCES IN SCHEMA graphrag TO %I', p_role);
    EXECUTE format('GRANT EXECUTE ON FUNCTION graphrag._evidence_acls(bigint[], bigint[]) TO %I', p_role);
    EXECUTE format('GRANT EXECUTE ON FUNCTION graphrag._may_supersede(bigint, text[]) TO %I', p_role);
    EXECUTE format('GRANT EXECUTE ON FUNCTION graphrag.community_context(bigint, integer) TO %I', p_role);
    EXECUTE format('GRANT EXECUTE ON FUNCTION graphrag.find_duplicates(text, double precision, double precision, integer) TO %I', p_role);
END
$$;

-- Maintenance functions return graph content without exposure accounting:
-- they are for the pipeline (grant_writer), not for end users.
REVOKE ALL ON FUNCTION graphrag.community_context(bigint, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION graphrag.find_duplicates(text, double precision, double precision, integer) FROM PUBLIC;
