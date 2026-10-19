//! In-database integration tests (`cargo pgrx test pg18`). Each test runs in
//! its own transaction, which is rolled back afterwards (roles included).

#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;
    use pgrx::JsonB;

    fn q<T: FromDatum + IntoDatum>(sql: &str) -> Option<T> {
        Spi::get_one::<T>(sql).unwrap_or_else(|e| panic!("{e}: {sql}"))
    }

    fn run(sql: &str) {
        Spi::run(sql).unwrap_or_else(|e| panic!("{e}: {sql}"));
    }

    fn chunk_of(external_id: &str) -> i64 {
        q::<i64>(&format!(
            "SELECT c.id FROM graphrag.chunk c JOIN graphrag.document d ON d.id = c.document_id WHERE d.external_id = '{external_id}'"
        ))
        .unwrap()
    }

    fn entity_id(name: &str) -> i64 {
        q::<i64>(&format!(
            "SELECT id FROM graphrag.entity WHERE canonical_key = graphrag.canonical_key('{name}')"
        ))
        .unwrap()
    }

    /// A small fictional Japanese / English corpus with one HR-restricted document.
    fn setup() {
        run("SELECT graphrag.create_graph('demo')");
        run("CREATE ROLE gr_hr NOLOGIN");
        let docs = [
            ("d1", "株式会社ミナト製作所は横浜市に本社を置く精密機器メーカーである。", "NULL"),
            ("d2", "ミナト製作所の代表取締役は山田花子である。山田花子は2020年に就任した。", "NULL"),
            ("d3", "横浜市は神奈川県の県庁所在地であり、港湾都市として知られる。", "NULL"),
            ("d4", "人事記録：山田花子は2025年に健康上の理由で休職した。", "ARRAY['gr_hr']"),
            ("d5", "Kestrel Labs, a startup based in Osaka, partners with Minato Seisakusho on sensor chips.", "NULL"),
        ];
        for (ext, content, acl) in docs {
            run(&format!(
                "SELECT graphrag.add_document('demo', content => '{content}', external_id => '{ext}', title => '{ext}', acl => {acl})"
            ));
        }
        let extractions = [
            (
                "d1",
                r#"{"entities":[{"name":"株式会社ミナト製作所","type":"organization","description":"横浜市に本社を置く精密機器メーカー"},
                                     {"name":"横浜市","type":"location","description":"ミナト製作所の本社所在地"}],
                       "relations":[{"source":"ミナト製作所","target":"横浜市","type":"headquartered in","description":"本社所在地"}]}"#,
            ),
            (
                "d2",
                r#"{"entities":[{"name":"山田花子","type":"person","description":"ミナト製作所の代表取締役"}],
                       "relationships":[{"src_id":"ミナト製作所(株)","tgt_id":"山田花子","relation":"LED_BY","description":"代表取締役","valid_from":"2020-04-01","exclusive":true}]}"#,
            ),
            (
                "d3",
                r#"{"entities":[{"name":"横浜市","type":"location","description":"神奈川県の県庁所在地、港湾都市"},
                                     {"name":"神奈川県","type":"location","description":"県庁所在地は横浜市"}],
                       "relations":[{"source":"横浜市","target":"神奈川県","type":"capital_of","description":"県庁所在地"}]}"#,
            ),
            (
                "d4",
                r#"{"entities":[{"name":"山田花子","type":"person","description":"2025年に健康上の理由で休職"},
                                     {"name":"休職","type":"status","description":"人事上の休職"}],
                       "relations":[{"source":"山田花子","target":"休職","type":"status","description":"2025年から休職中","valid_from":"2025-01-01"}]}"#,
            ),
            (
                "d5",
                r#"{"entities":[{"entity_name":"Kestrel Labs","entity_type":"organization","description":"Osaka-based startup"},
                                     {"entity_name":"Osaka","entity_type":"location"},
                                     {"entity_name":"Minato Seisakusho","entity_type":"organization","description":"sensor chip partner"}],
                       "relationships":[{"src_id":"Kestrel Labs","tgt_id":"Osaka","relationship":"based_in","weight":"2"},
                                        {"src_id":"Kestrel Labs","tgt_id":"Minato Seisakusho","relationship":"partners_with","description":"sensor chips"}]}"#,
            ),
        ];
        for (ext, json) in extractions {
            let id = chunk_of(ext);
            let res = q::<JsonB>(&format!(
                "SELECT graphrag.ingest_extraction({id}, '{}'::jsonb)",
                json.replace('\'', "''")
            ))
            .unwrap();
            assert_eq!(res.0["skipped"], 0, "{ext}: {:?}", res.0);
        }
    }

    #[pg_test]
    fn canonical_keys_dedupe_entities() {
        setup();
        // 株式会社ミナト製作所 / ミナト製作所 / ミナト製作所(株) are one entity with aliases.
        assert_eq!(
            q::<i64>("SELECT count(*) FROM graphrag.entity WHERE canonical_key = 'ミナト製作所'"),
            Some(1)
        );
        let aliases = q::<Vec<String>>(
            "SELECT aliases FROM graphrag.entity WHERE canonical_key = 'ミナト製作所'",
        )
        .unwrap();
        assert!(aliases.iter().any(|a| a.contains("(株)")), "{aliases:?}");
        assert_eq!(q::<String>("SELECT rel_type FROM graphrag.fact WHERE target = '横浜市' AND source LIKE '%ミナト%'"), Some("headquartered_in".into()));
        assert_eq!(
            q::<String>("SELECT entity_type FROM graphrag.entity WHERE name = '山田花子'"),
            Some("PERSON".into())
        );
        // provenance
        assert_eq!(q::<i64>("SELECT count(*) FROM graphrag.mention WHERE entity_id = (SELECT id FROM graphrag.entity WHERE name = '山田花子')"), Some(2));
        assert_eq!(
            q::<i64>("SELECT count(*) FROM graphrag.relation_evidence"),
            Some(6)
        );
        // chunks get lexemes and PG18 uuidv7 ids
        assert_eq!(
            q::<bool>(
                "SELECT bool_and(lexemes IS NOT NULL AND token_count > 0) FROM graphrag.chunk"
            ),
            Some(true)
        );
        assert_eq!(
            q::<i32>("SELECT uuid_extract_version(uid) FROM graphrag.document LIMIT 1"),
            Some(7)
        );
    }

    #[pg_test]
    fn multi_hop_local_search_via_ppr() {
        setup();
        // The question never mentions 横浜市 or 神奈川県: those chunks are reached
        // through 山田花子 -> ミナト製作所 -> 横浜市 -> 神奈川県.
        let top = q::<Vec<String>>(
            "SELECT array_agg(d.external_id ORDER BY s.rank)
               FROM graphrag.local_search('demo', query => '山田花子が率いる会社の本社がある都市はどこの県？', top_k => 5) s
               JOIN graphrag.document d ON d.id = s.document_id",
        )
        .unwrap();
        assert!(
            top.contains(&"d1".to_string()) && top.contains(&"d3".to_string()),
            "{top:?}"
        );
        assert!(
            !top.contains(&"d5".to_string()),
            "unrelated document retrieved: {top:?}"
        );
        // Graph-only ranking follows graph distance from the seed.
        let graph_only = q::<Vec<String>>(
            "SELECT array_agg(d.external_id ORDER BY s.rank)
               FROM graphrag.local_search('demo', seed_entities => ARRAY['山田花子'], top_k => 5, options => '{\"lexical\": false}') s
               JOIN graphrag.document d ON d.id = s.document_id",
        )
        .unwrap();
        let pos = |x: &str| graph_only.iter().position(|d| d == x).unwrap_or(usize::MAX);
        assert!(
            pos("d2") < pos("d1") && pos("d1") < pos("d3") && pos("d3") < usize::MAX,
            "{graph_only:?}"
        );
        assert_eq!(pos("d5"), usize::MAX, "{graph_only:?}");
        let ranked = q::<Vec<String>>("SELECT array_agg(name ORDER BY score DESC) FROM graphrag.rank_entities('demo', seed_entities => ARRAY['山田花子'], top_k => 5)").unwrap();
        assert_eq!(ranked[0], "山田花子");
        assert!(ranked.iter().any(|n| n.contains("ミナト")), "{ranked:?}");
        // graph_score is reported for graph-retrieved chunks
        assert_eq!(
            q::<bool>("SELECT bool_or(graph_score IS NOT NULL) FROM graphrag.local_search('demo', seed_entities => ARRAY['山田花子'], top_k => 5)"),
            Some(true)
        );
    }

    #[pg_test]
    fn relational_paths_are_explainable() {
        setup();
        let path = q::<String>("SELECT path FROM graphrag.relational_paths('demo', ARRAY['山田花子', '神奈川県'], max_hops => 3) WHERE rank = 1").unwrap();
        assert_eq!(path, "山田花子 <-[led_by]- 株式会社ミナト製作所 -[headquartered_in]-> 横浜市 -[capital_of]-> 神奈川県");
        let ev = q::<Vec<i64>>("SELECT evidence_chunk_ids FROM graphrag.relational_paths('demo', ARRAY['山田花子', '神奈川県']) WHERE rank = 1").unwrap();
        assert_eq!(ev.len(), 3);
    }

    #[pg_test]
    fn japanese_lexical_search() {
        setup();
        let first = q::<i64>(
            "SELECT chunk_id FROM graphrag.lexical_search('demo', '神奈川県の県庁所在地') LIMIT 1",
        )
        .unwrap();
        assert_eq!(first, chunk_of("d3"));
        let en = q::<i64>(
            "SELECT chunk_id FROM graphrag.lexical_search('demo', 'sensor chips startup') LIMIT 1",
        )
        .unwrap();
        assert_eq!(en, chunk_of("d5"));
    }

    #[pg_test]
    fn acl_hides_restricted_evidence_and_reports() {
        setup();
        run("CREATE ROLE gr_alice NOLOGIN");
        run("CREATE ROLE gr_bob NOLOGIN IN ROLE gr_hr");
        run("SELECT graphrag.grant_reader('gr_alice'), graphrag.grant_reader('gr_bob')");
        let hanako = entity_id("山田花子");
        run(&format!("SELECT graphrag.set_report('entity', {hanako}, '山田花子は代表取締役。2025年から休職中。', generated_by => 'test-model')"));
        assert_eq!(
            q::<JsonB>(&format!(
                "SELECT required_acls FROM graphrag.report WHERE entity_id = {hanako}"
            ))
            .unwrap()
            .0,
            serde_json::json!([["gr_hr"]])
        );

        let ctx_sql = "SELECT (graphrag.build_context('demo', query => '山田花子の状況は？', max_tokens => 2000))->>'context'";
        let search_sql = "SELECT coalesce(string_agg(content, ' '), '') FROM graphrag.local_search('demo', query => '山田花子 休職', top_k => 10)";
        let paths_sql =
            "SELECT count(*) FROM graphrag.relational_paths('demo', ARRAY['山田花子', '休職'])";

        for rls in [false, true] {
            if rls {
                run("SELECT graphrag.enable_row_security()");
            }
            run("SET ROLE gr_alice");
            let ctx = q::<String>(ctx_sql).unwrap();
            assert!(
                !ctx.contains("休職"),
                "rls={rls}: alice must not see HR facts:\n{ctx}"
            );
            assert!(ctx.contains("代表取締役"), "rls={rls}: {ctx}");
            assert!(
                !q::<String>(search_sql).unwrap().contains("休職"),
                "rls={rls}"
            );
            assert_eq!(q::<i64>(paths_sql), Some(0), "rls={rls}");
            if rls {
                // Direct SQL is filtered too.
                assert_eq!(
                    q::<i64>("SELECT count(*) FROM graphrag.chunk WHERE content LIKE '%休職%'"),
                    Some(0)
                );
                assert_eq!(
                    q::<i64>("SELECT count(*) FROM graphrag.entity WHERE name = '休職'"),
                    Some(0)
                );
                assert_eq!(q::<i64>("SELECT count(*) FROM graphrag.report"), Some(0));
            }
            run("RESET ROLE");

            run("SET ROLE gr_bob");
            let ctx = q::<String>(ctx_sql).unwrap();
            assert!(ctx.contains("休職"), "rls={rls}: bob is in gr_hr:\n{ctx}");
            assert!(
                q::<String>(search_sql).unwrap().contains("休職"),
                "rls={rls}"
            );
            assert_eq!(q::<i64>(paths_sql), Some(1), "rls={rls}");
            run("RESET ROLE");
        }
    }

    #[pg_test]
    fn temporal_facts_and_exclusive_relations() {
        setup();
        run("SELECT graphrag.upsert_relation('demo', 'ミナト製作所', '鈴木一郎', 'led_by', valid_from => '2026-04-01', exclusive => true, target_type => 'person')");
        // The previous CEO's term was closed, not deleted.
        assert_eq!(
            q::<String>("SELECT upper(valid_during)::date::text FROM graphrag.relation r JOIN graphrag.entity t ON t.id = r.target_id WHERE t.name = '山田花子' AND r.rel_type = 'led_by'"),
            Some("2026-04-01".into())
        );
        let who = |as_of: &str| {
            q::<String>(&format!(
                "SELECT string_agg(target, ',') FROM graphrag.fact
                  WHERE rel_type = 'led_by' AND tstzrange(valid_from, valid_to) @> '{as_of}'::timestamptz"
            ))
        };
        assert_eq!(who("2023-01-01"), Some("山田花子".into()));
        assert_eq!(who("2026-10-01"), Some("鈴木一郎".into()));
        // Retrieval honours as_of.
        let paths = |as_of: &str| {
            q::<i64>(&format!(
                "SELECT count(*) FROM graphrag.relational_paths('demo', ARRAY['ミナト製作所', '鈴木一郎'], as_of => '{as_of}')"
            ))
            .unwrap()
        };
        assert_eq!(paths("2023-01-01"), 0);
        assert_eq!(paths("2026-10-01"), 1);
        // PostgreSQL 18 WITHOUT OVERLAPS forbids two overlapping versions of one fact.
        run(
            "DO $$ BEGIN
                INSERT INTO graphrag.relation (graph_id, source_id, target_id, rel_type, valid_during)
                SELECT graph_id, source_id, target_id, rel_type, tstzrange('2027-01-01', NULL)
                  FROM graphrag.relation WHERE rel_type = 'led_by' AND upper_inf(valid_during);
                RAISE EXCEPTION 'overlap was accepted';
             EXCEPTION WHEN exclusion_violation OR unique_violation THEN NULL;
             END $$",
        );
        // Manual invalidation
        let rid = q::<i64>("SELECT id FROM graphrag.fact WHERE rel_type = 'capital_of'").unwrap();
        assert_eq!(
            q::<bool>(&format!(
                "SELECT graphrag.invalidate_relation({rid}, '2030-01-01')"
            )),
            Some(true)
        );
        assert_eq!(
            q::<bool>(&format!(
                "SELECT invalidated_at IS NOT NULL FROM graphrag.relation WHERE id = {rid}"
            )),
            Some(true)
        );
    }

    #[pg_test]
    fn forget_document_erases_derived_knowledge() {
        setup();
        let hanako = entity_id("山田花子");
        run(&format!(
            "SELECT graphrag.set_report('entity', {hanako}, '休職中の代表取締役')"
        ));
        let report = q::<JsonB>("SELECT graphrag.forget_document('demo', 'd4')")
            .unwrap()
            .0;
        assert_eq!(report["chunks_deleted"], 1, "{report}");
        assert_eq!(report["entities_deleted"], 1, "{report}"); // 休職
        assert_eq!(report["relations_deleted"], 1, "{report}"); // 山田花子 -status-> 休職
        assert_eq!(report["reports_deleted"], 1, "{report}");
        assert_eq!(
            q::<i64>("SELECT count(*) FROM graphrag.entity WHERE name = '休職'"),
            Some(0)
        );
        assert_eq!(
            q::<i64>("SELECT count(*) FROM graphrag.entity WHERE name = '山田花子'"),
            Some(1)
        );
        assert_eq!(q::<i64>("SELECT count(*) FROM graphrag.mention m JOIN graphrag.chunk c ON c.id = m.chunk_id WHERE c.content LIKE '%休職%'"), Some(0));
        assert_eq!(
            q::<i64>("SELECT count(*) FROM graphrag.erasure_log"),
            Some(1)
        );
        assert_eq!(
            q::<i64>("SELECT count(*) FROM graphrag.pending_report WHERE kind = 'entity'"),
            Some(0)
        );
    }

    #[pg_test]
    fn communities_reports_and_global_search() {
        setup();
        let res = q::<JsonB>("SELECT graphrag.build_communities('demo', max_cluster_size => 3)")
            .unwrap()
            .0;
        assert!(res["communities"].as_i64().unwrap() >= 2, "{res}");
        assert_eq!(res["reports_reused"], 0);
        let pending =
            q::<i64>("SELECT count(*) FROM graphrag.pending_report WHERE kind = 'community'")
                .unwrap();
        assert_eq!(pending, res["communities"].as_i64().unwrap());
        // The community that contains 横浜市 gets an LLM report.
        let cid = q::<i64>(
            "SELECT c.id FROM graphrag.community c JOIN graphrag.community_member cm ON cm.community_id = c.id
               JOIN graphrag.entity e ON e.id = cm.entity_id WHERE e.name = '横浜市' ORDER BY c.level DESC LIMIT 1",
        )
        .unwrap();
        let ctx = q::<String>(&format!("SELECT graphrag.community_context({cid})")).unwrap();
        assert!(ctx.contains("横浜市"), "{ctx}");
        run(&format!("SELECT graphrag.set_report('community', {cid}, '横浜を拠点とする企業と自治体のまとまり', title => '横浜クラスタ', embedding => ARRAY[1,0,0]::real[], generated_by => 'test')"));
        let hit = q::<String>("SELECT title FROM graphrag.global_search('demo', query_embedding => ARRAY[1,0,0]::real[], top_k => 1)").unwrap();
        assert_eq!(hit, "横浜クラスタ");
        // Rebuilding with an unchanged graph re-uses the report.
        let res2 = q::<JsonB>("SELECT graphrag.build_communities('demo', max_cluster_size => 3)")
            .unwrap()
            .0;
        assert_eq!(res2["reports_reused"], 1, "{res2}");
        assert_eq!(
            q::<bool>("SELECT stale FROM graphrag.report WHERE community_id IS NOT NULL"),
            Some(false)
        );
        // New evidence inside the community marks its report stale.
        let c1 = chunk_of("d3");
        run(&format!("SELECT graphrag.upsert_relation('demo', '横浜市', '神奈川県', 'located_in', chunk_id => {c1})"));
        assert_eq!(
            q::<bool>("SELECT stale FROM graphrag.report WHERE community_id IS NOT NULL"),
            Some(true)
        );
    }

    #[pg_test]
    fn global_search_redacts_reports_built_on_restricted_evidence() {
        setup();
        run("CREATE ROLE gr_carol NOLOGIN");
        run("SELECT graphrag.grant_reader('gr_carol')");
        run("SELECT graphrag.build_communities('demo', max_cluster_size => 100)");
        let cid = q::<i64>(
            "SELECT cm.community_id FROM graphrag.community_member cm JOIN graphrag.entity e ON e.id = cm.entity_id WHERE e.name = '休職'",
        )
        .unwrap();
        run(&format!("SELECT graphrag.set_report('community', {cid}, '代表取締役の休職を含むクラスタ', title => '経営体制')"));
        run("SET ROLE gr_carol");
        let row = q::<String>(&format!(
            "SELECT visibility || ':' || coalesce(summary, '') || ':' || array_to_string(members, ',')
               FROM graphrag.global_search('demo', query => '経営体制') WHERE community_id = {cid}"
        ))
        .unwrap();
        run("RESET ROLE");
        assert!(row.starts_with("redacted::"), "{row}");
        assert!(!row.contains("休職"), "{row}");
        assert!(row.contains("山田花子"), "{row}");
    }

    #[pg_test]
    fn duplicates_and_merge() {
        setup();
        run("SELECT graphrag.upsert_entity('demo', 'Kestrel Lab', 'organization', chunk_id => (SELECT c.id FROM graphrag.chunk c JOIN graphrag.document d ON d.id = c.document_id WHERE d.external_id = 'd5'))");
        let pair = q::<String>("SELECT name_a || '|' || name_b FROM graphrag.find_duplicates('demo', threshold => 0.85) LIMIT 1").unwrap();
        assert!(
            pair.contains("Kestrel Lab|") || pair.contains("|Kestrel Lab"),
            "{pair}"
        );
        // Cross-script duplicate (romaji vs. katakana) merged by hand.
        let keep = entity_id("ミナト製作所");
        let dup = entity_id("Minato Seisakusho");
        let res = q::<JsonB>(&format!(
            "SELECT graphrag.merge_entities({keep}, ARRAY[{dup}])"
        ))
        .unwrap()
        .0;
        assert_eq!(res["relations_moved"], 1, "{res}");
        assert_eq!(
            q::<i64>(&format!(
                "SELECT count(*) FROM graphrag.entity WHERE id = {dup}"
            )),
            Some(0)
        );
        assert_eq!(
            q::<i64>("SELECT count(*) FROM graphrag.fact WHERE source = 'Kestrel Labs' AND target = '株式会社ミナト製作所'"),
            Some(1)
        );
        assert_eq!(
            q::<bool>(&format!(
                "SELECT 'Minato Seisakusho' = ANY(aliases) FROM graphrag.entity WHERE id = {keep}"
            )),
            Some(true)
        );
        // After the merge, Kestrel Labs is two hops from 横浜市.
        let path = q::<String>("SELECT path FROM graphrag.relational_paths('demo', ARRAY['Kestrel Labs', '横浜市']) WHERE rank = 1").unwrap();
        assert!(path.contains("株式会社ミナト製作所"), "{path}");
    }

    #[pg_test]
    fn lazy_index_without_llm() {
        run("SELECT graphrag.create_graph('lazy')");
        run("SELECT graphrag.add_document('lazy', content => 'GraphRAGはMicrosoft Researchが提案した手法である。GraphRAGはコミュニティ要約を使う。', external_id => 'a')");
        run("SELECT graphrag.add_document('lazy', content => 'HippoRAGはPersonalized PageRankで多段推論を行う。Personalized PageRankは知識グラフ上で動く。', external_id => 'b')");
        run("SELECT graphrag.add_document('lazy', content => 'LightRAGとHippoRAGはどちらも知識グラフを利用する。', external_id => 'c')");
        let res = q::<JsonB>("SELECT graphrag.lazy_index('lazy')").unwrap().0;
        assert_eq!(res["chunks"], 3, "{res}");
        assert_eq!(res["llm_tokens"], 0);
        assert!(q::<i64>("SELECT count(*) FROM graphrag.relation r JOIN graphrag.graph g ON g.id = r.graph_id WHERE g.name = 'lazy' AND rel_type = 'co_occurs'").unwrap() > 3);
        // Second run is a no-op (chunks already indexed).
        assert_eq!(
            q::<JsonB>("SELECT graphrag.lazy_index('lazy')").unwrap().0["chunks"],
            0
        );
        // Query about LightRAG reaches the PageRank chunk through HippoRAG.
        let top = q::<Vec<String>>(
            "SELECT array_agg(d.external_id ORDER BY s.rank) FROM graphrag.local_search('lazy', query => 'LightRAGに関連する研究は？', top_k => 3,
                     options => '{\"lexical\": false}') s JOIN graphrag.document d ON d.id = s.document_id",
        )
        .unwrap();
        assert_eq!(top[0], "c", "{top:?}");
        assert!(top.contains(&"b".to_string()), "{top:?}");
    }

    #[pg_test]
    fn dense_embeddings_seed_and_rank() {
        setup();
        // Toy 3-d embeddings: d3 and 神奈川県 point the same way as the query.
        run("UPDATE graphrag.chunk SET embedding = ARRAY[0, 1, 0]::real[]");
        run(&format!(
            "UPDATE graphrag.chunk SET embedding = ARRAY[1, 0, 0]::real[] WHERE id = {}",
            chunk_of("d3")
        ));
        run("UPDATE graphrag.entity SET embedding = ARRAY[0, 0, 1]::real[]");
        run("UPDATE graphrag.entity SET embedding = ARRAY[1, 0.1, 0]::real[] WHERE name = '神奈川県'");
        let first = q::<i64>(
            "SELECT chunk_id FROM graphrag.local_search('demo', query_embedding => ARRAY[1, 0, 0]::real[], top_k => 3) ORDER BY rank LIMIT 1",
        )
        .unwrap();
        assert_eq!(first, chunk_of("d3"));
        let seed = q::<String>(
            "SELECT name FROM graphrag.rank_entities('demo', query_embedding => ARRAY[1, 0, 0]::real[],
                    options => '{\"embedding_seed_k\": 1}') WHERE seed",
        );
        assert_eq!(seed, Some("神奈川県".into()));
        // Mismatched dimensions are ignored rather than failing.
        assert_eq!(
            q::<i64>("SELECT count(*) FROM graphrag.local_search('demo', query_embedding => ARRAY[1, 0]::real[], query => 'xyz')"),
            Some(0)
        );
    }

    #[pg_test]
    fn document_acl_change_reaches_reports() {
        setup();
        let company = entity_id("ミナト製作所");
        run(&format!(
            "SELECT graphrag.set_report('entity', {company}, '横浜の精密機器メーカー')"
        ));
        assert_eq!(
            q::<JsonB>(&format!(
                "SELECT required_acls FROM graphrag.report WHERE entity_id = {company}"
            ))
            .unwrap()
            .0,
            serde_json::json!([])
        );
        // Tightening the ACL of d1 propagates to its chunk and to the report.
        run("UPDATE graphrag.document SET acl = ARRAY['gr_hr'] WHERE external_id = 'd1'");
        assert_eq!(
            q::<Vec<String>>(&format!(
                "SELECT acl FROM graphrag.chunk WHERE id = {}",
                chunk_of("d1")
            )),
            Some(vec!["gr_hr".into()])
        );
        let row = q::<String>(&format!("SELECT required_acls::text || ':' || stale FROM graphrag.report WHERE entity_id = {company}")).unwrap();
        assert_eq!(row, r#"[["gr_hr"]]:true"#);
    }

    #[pg_test]
    fn build_context_is_cited_and_budgeted() {
        setup();
        let ctx = q::<JsonB>("SELECT graphrag.build_context('demo', query => '山田花子が率いる会社の本社は？', max_tokens => 400)").unwrap().0;
        let text = ctx["context"].as_str().unwrap();
        assert!(text.contains("## Entities"), "{text}");
        assert!(text.contains("## Sources"), "{text}");
        assert!(text.contains("[#"), "{text}");
        assert!(ctx["token_estimate"].as_u64().unwrap() <= 400, "{ctx}");
        assert!(
            ctx["seeds"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s["name"] == "山田花子"),
            "{ctx}"
        );
        let sources = ctx["sections"]["sources"].as_array().unwrap();
        assert!(!sources.is_empty());
        // Reasoning paths between seeds appear when there are at least two.
        let ctx = q::<JsonB>("SELECT graphrag.build_context('demo', seed_entities => ARRAY['山田花子', '神奈川県'], max_tokens => 2000)").unwrap().0;
        assert!(
            ctx["context"]
                .as_str()
                .unwrap()
                .contains("## Reasoning paths"),
            "{ctx}"
        );
    }

    #[pg_test]
    fn exposure_budget_limits_graph_extraction() {
        setup();
        run("CREATE ROLE gr_eve NOLOGIN");
        run("SELECT graphrag.grant_reader('gr_eve')");
        run("SET graphrag.exposure_tracking = on");
        run("SET graphrag.exposure_budget = 0.4");
        let total = q::<i64>("SELECT count(*) FROM graphrag.entity").unwrap() as f64;
        run("SET ROLE gr_eve");
        for name in [
            "山田花子",
            "横浜市",
            "神奈川県",
            "Kestrel Labs",
            "Osaka",
            "ミナト製作所",
        ] {
            run(&format!("SELECT count(*) FROM graphrag.rank_entities('demo', seed_entities => ARRAY['{name}'], top_k => 50)"));
        }
        let seen = q::<i64>("SELECT count(*) FROM graphrag.exposure").unwrap() as f64;
        run("RESET ROLE");
        assert!(seen / total <= 0.4 + 1e-9, "{seen}/{total}");
        assert!(seen > 0.0);
        let ratio = q::<f64>(
            "SELECT ratio FROM graphrag.exposure_report('demo') WHERE principal = 'gr_eve'",
        )
        .unwrap();
        assert!(ratio <= 0.4 + 1e-9);
        assert!(q::<i64>("SELECT graphrag.reset_exposure('demo', 'gr_eve')").unwrap() > 0);
    }

    #[pg_test]
    fn chunking_and_text_functions() {
        assert_eq!(
            q::<String>("SELECT graphrag.canonical_key('トヨタ自動車株式会社')"),
            Some("トヨタ自動車".into())
        );
        assert_eq!(q::<i64>("SELECT count(*) FROM graphrag.chunk_text(repeat('これは文です。', 100), max_tokens => 50, overlap_tokens => 0)"), Some(15));
        assert_eq!(q::<String>("SELECT phrase FROM graphrag.extract_keyphrases('PostgreSQL拡張でGraphRAGを実装する。GraphRAGは便利。') LIMIT 1"), Some("GraphRAG".into()));
        assert!(
            q::<f64>(
                "SELECT graphrag.cosine_similarity(ARRAY[1,2,3]::real[], ARRAY[1,2,3]::real[])"
            )
            .unwrap()
                > 0.999
        );
        run("SELECT graphrag.create_graph('empty')");
        assert_eq!(
            q::<JsonB>("SELECT graphrag.stats('empty')").unwrap().0["entities"],
            0
        );
    }
}
