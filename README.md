# pg_graphrag — PostgreSQL 18 のための GraphRAG エンジン（Rust / pgrx）

`pg_graphrag` は、GraphRAG（グラフ構造を使う検索拡張生成）に必要な **知識グラフの保存・検索・統治** を PostgreSQL の中で完結させる拡張機能です。Rust（[pgrx](https://github.com/pgcentralfoundation/pgrx) 0.19）で実装し、PostgreSQL 18 の新機能（`uuidv7()`、時間制約 `WITHOUT OVERLAPS`）を使っています。

LLM の呼び出し（抽出・要約・回答生成）はアプリケーション側に残し、データベースは次を担当します。

- **出典付きの知識グラフ**: 文書 → チャンク → エンティティ／関係。すべての事実がどのチャンクに由来するかを保持
- **グラフアルゴリズム**: ローカルプッシュ型 Personalized PageRank（HippoRAG 系の多段推論）、階層的 Leiden（Microsoft GraphRAG 系のコミュニティ要約）、フロー枝刈りによる関係パス探索（PathRAG）
- **日本語対応の検索**: NFKC 正規化と法人格除去によるエンティティ名寄せ、CJK バイグラム BM25、LLM 不要のキーフレーズ抽出
- **権限・時間・削除の統治**: 根拠ベースのアクセス制御、時間付きの事実、忘れられる権利への対応、グラフ抽出攻撃への露出バジェット

最終的に `graphrag.build_context()` が、**呼び出したロールが読んでよい根拠だけ** から、トークン予算内で、出典マーカー `[#chunk_id]` 付きのプロンプト用コンテキストを返します。

```sql
SELECT graphrag.build_context('demo', query => '山田花子が率いる会社の本社がある県は？', max_tokens => 600)->>'context';
```
```text
## Entities
- 山田花子 (PERSON): ミナト製作所の代表取締役
- 株式会社ミナト製作所 (ORGANIZATION): 横浜市に本社を置く精密機器メーカー
- 横浜市 (LOCATION): 神奈川県の県庁所在地
...
## Relationships
- 株式会社ミナト製作所 -[led_by]-> 山田花子 (valid 2020-04-01 – now): 代表取締役 [#2]
- 株式会社ミナト製作所 -[headquartered_in]-> 横浜市: 本社所在地 [#1]
- 横浜市 -[capital_of]-> 神奈川県: 県庁所在地 [#3]

## Sources
[#2] (役員情報) ミナト製作所の代表取締役は山田花子である。…
[#1] (会社概要) 株式会社ミナト製作所は横浜市に本社を置く精密機器メーカーである。
[#3] (自治体) 横浜市は神奈川県の県庁所在地であり、港湾都市として知られる。
```

質問文には「横浜」も「神奈川」も出てきませんが、`山田花子 → ミナト製作所 → 横浜市 → 神奈川県` というグラフ上の伝播で、答えを含むチャンクに到達しています。

---

## 目次

1. [背景：GraphRAG の現状と社会的な課題](#1-背景graphrag-の現状と社会的な課題)
2. [課題と設計の対応](#2-課題と設計の対応)
3. [研究との対応](#3-研究との対応)
4. [アーキテクチャ](#4-アーキテクチャ)
5. [ビルドとインストール](#5-ビルドとインストール)
6. [使い方](#6-使い方)
7. [API リファレンス](#7-api-リファレンス)
8. [セキュリティモデル](#8-セキュリティモデル)
9. [設定パラメータ](#9-設定パラメータ)
10. [テスト](#10-テスト)
11. [制約と今後の課題](#11-制約と今後の課題)
12. [参考文献](#12-参考文献)

---

## 1. 背景：GraphRAG の現状と社会的な課題

ベクトル検索だけの RAG は、複数の文書をまたいでたどる質問（多段推論）や、コーパス全体についての質問（「主なテーマは何か」）が苦手です。GraphRAG は文書から知識グラフを作ってこれを補います。代表的な系統は次のとおりです。

| 系統 | 代表例 | 考え方 |
|---|---|---|
| コミュニティ要約型 | Microsoft GraphRAG (2024) | LLM で抽出したグラフを Leiden で階層的にクラスタリングし、各コミュニティの要約から全体的な質問に答える |
| グラフ伝播型 | HippoRAG (2024) / HippoRAG 2 (2025) | 質問中のエンティティから Personalized PageRank を走らせ、多段の関連を 1 回の検索で拾う |
| 軽量・増分型 | LightRAG (2024)、LazyGraphRAG (2024)、LinearRAG (2025)、TIGRAG (2026) | LLM による索引コストを下げる。関係抽出を省く、名詞句の共起だけでグラフを作る、などの方法をとる |
| パス型 | PathRAG (2025) | 冗長な部分グラフではなく、信頼度の高い関係パスだけをプロンプトに入れる |
| 時間付きグラフ | Zep / Graphiti (2025) | 事実に有効期間を持たせ、矛盾する新事実が来たら古い辺を削除せず無効化する |

研究が進む一方で、実運用では次の問題が目立っています。

1. **権限漏洩と抽出攻撃**: コミュニティ要約は複数の文書を混ぜて作るため、閲覧権限の異なる文書の内容が 1 つの要約に入り得ます。Graph RAG は生テキストの漏洩は減らすものの、エンティティや関係といった構造情報は抜き出されやすいことが示されています（Liu et al., ACL 2026 Findings）。ブラックボックスの問い合わせだけで知識グラフの 9 割以上を再構成した報告もあります（GraphSteal, 2026）。
2. **削除の権利**: GDPR 17 条や個人情報保護法の削除要求に応えるには、元文書だけでなく、そこから派生したエンティティ説明・LLM 要約・埋め込みまで消す必要があります。出典を追跡していないと、どこまで消せばよいかが分かりません。
3. **知識の鮮度**: 社長交代や所在地変更のように、事実には有効期間があります。上書きすると履歴が失われ、残しておくと矛盾します。
4. **索引コスト**: LLM による抽出と要約はコーパス全体に対して走り、文書が変わるたびに再実行されがちです。運用コストの大半がここに集中します。
5. **ハルシネーションと説明責任**: 回答が根拠に基づいているかを利用者や監査者が確かめられる必要があります。EU AI 法の透明性・記録保持の要件も、この方向を求めています。
6. **日本語**: 全角と半角、`株式会社` と `(株)` の表記ゆれでグラフが分断されます。また PostgreSQL 標準の全文検索は日本語を分かち書きしません。
7. **データのサイロ化**: ベクトル DB、グラフ DB、RDB を別々に運用すると、同期・整合性・権限管理が三重になります。

## 2. 課題と設計の対応

| 課題 | pg_graphrag の設計 |
|---|---|
| 権限漏洩 | **根拠ベースの可視性**: チャンクの ACL（PostgreSQL ロール名）から、エンティティ・関係・グラフ探索の可視性を導出します。LLM が書いた要約には、生成に使った全根拠の ACL 集合（`required_acls`）を記録し、そのすべてを満たすロールにだけ見せます。満たさない場合は要約を伏せ、見えるメンバーだけを返します（`visibility = 'redacted'`）。抽出結果がエンティティについて述べた内容（別名・説明・埋め込み・metadata）は言及の側に保存し、根拠のチャンクと同じ ACL で見せます。制限付き文書から来た排他的な事実は、それを読めないロールにも見えている事実を閉じません。`enable_row_security()` で直接の SQL アクセスにも RLS をかけられます |
| 抽出攻撃 | **露出バジェット**: `graphrag.exposure_tracking` / `exposure_budget` を設定すると、ロールごとに開示済みのエンティティを記録し、グラフ全体に対する開示割合が上限を超えた時点で新しいエンティティを含む結果を返さなくなります。設定は superuser だけが変更できます |
| 削除の権利 | **出典追跡による削除**: `forget_document()` は文書・チャンク・言及・根拠を削除し、根拠がなくなった事実と、削除内容を含み得る LLM 要約を消し、関連する埋め込みをクリアして、`erasure_log` に内容を含まない監査記録を残します。呼び出し側に RLS で見えない派生物も含めて消します |
| 知識の鮮度 | **時間付きの事実**: 関係は有効期間 `valid_during tstzrange` を持ちます。PostgreSQL 18 の `UNIQUE (…, valid_during WITHOUT OVERLAPS)` が「同じ事実の有効期間は重ならない」ことを DB 制約として保証します。`exclusive => true` で登録すると、同じ主語・述語の現行事実を新事実の開始時点で閉じます（削除はしません）。すべての検索関数は `as_of` を受け取ります |
| 索引コスト | **`lazy_index()`**: LLM を使わずにキーフレーズの共起グラフを作ります（LLM トークン 0）。**要約の再利用**: `build_communities()` はメンバー構成が変わらないコミュニティの要約を引き継ぎ、根拠が変わった要約はトリガで `stale` にします。`pending_report` ビューに出たものだけを LLM に送れば済みます |
| 説明責任 | すべての事実に出典チャンクを紐付けます。コンテキストには `[#chunk_id]` 付きの出典が入り、`relational_paths()` は推論パスと根拠チャンクを返します。要約には `generated_by`（生成モデル名）を記録します |
| 日本語 | `canonical_key()` で NFKC、小文字化、区切り記号の除去、`株式会社` / `(株)` / `㈱` / `Inc.` などの法人格除去を行います。`lexical_tokens()` は CJK を文字バイグラムに分解して GIN 索引付きの BM25 検索を可能にし、`extract_keyphrases()` は漢字・カタカナ（＋隣接英数字）の連続から名詞句を推定します |
| サイロ化 | 文書・ベクトル（`real[]`）・グラフ・要約・権限・監査記録が 1 つの PostgreSQL のトランザクション内に収まります |

## 3. 研究との対応

| 手法 | 論文 | 実装 |
|---|---|---|
| Personalized PageRank による多段検索 | HippoRAG (NeurIPS 2024)、HippoRAG 2 (2025) | `rank_entities`、`local_search`。質問中のエンティティを種にし、言及チャンク数でノード特異性の重みを付け、チャンクには含むエンティティの PPR 質量でスコアを付けます |
| ローカルプッシュ PPR | Andersen, Chung, Lang (FOCS 2006) | `ppr.rs`。残差が閾値を超えたノードの隣接リストだけを SPI で遅延・一括取得するため、グラフ全体を読みません。その場で権限と有効期間を適用します |
| コミュニティ要約とグローバル検索 | Edge et al., Microsoft GraphRAG (2024) | `build_communities`、`community_context`、`set_report`、`global_search` |
| Leiden 法 | Traag, Waltman, van Eck (Sci. Rep. 2019) | `leiden.rs`。移動・細分化・集約の 3 段で、コミュニティの連結性を保証します。Microsoft GraphRAG と同じく、上限サイズを超えるコミュニティを再分割して階層を作ります |
| 動的なコミュニティ選択 | Microsoft GraphRAG (2024) | `global_search(level => NULL)` は全階層から関連度順に選び、祖先と子孫を同時には返しません |
| 関係パスとフロー枝刈り | PathRAG (2025) | `relational_paths`、`paths.rs`。資源が `decay / 次数` で減衰し閾値未満で枝刈りされるため、ハブ経由のパスは自然に除かれます。信頼度の高いパスはプロンプトの末尾（質問の直前）に置きます |
| LLM を使わない索引 | LazyGraphRAG (2024)、LinearRAG (2025)、TIGRAG (2026) | `lazy_index`、`extract_keyphrases` |
| 時間付き知識グラフと辺の無効化 | Zep / Graphiti (2025) | `upsert_relation(exclusive => true)`、`invalidate_relation`、`as_of` |
| ハイブリッド検索の統合 | Reciprocal Rank Fusion (Cormack et al., SIGIR 2009) | `local_search` は PPR・密ベクトル・BM25 の 3 系統を重み付き RRF で統合します |
| Graph RAG への抽出攻撃 | Liu et al. (ACL 2026 Findings)、GraphSteal (2026) | 露出バジェット、根拠ベース可視性、要約の伏せ字化 |

## 4. アーキテクチャ

```mermaid
flowchart LR
  subgraph App["アプリケーション（任意の LLM）"]
    X[抽出プロンプト] --> I
    S[要約プロンプト]
    A[回答生成]
  end
  subgraph PG["PostgreSQL 18 + pg_graphrag (Rust)"]
    I[ingest_extraction / lazy_index] --> KG[(document / chunk / entity / mention<br/>relation[valid_during] / relation_evidence)]
    KG --> L[build_communities<br/>階層 Leiden]
    L --> C[(community / report<br/>required_acls, stale)]
    C -- pending_report --> S
    S -- set_report --> C
    Q[build_context] --> PPR[local push PPR]
    Q --> BM[CJK bigram BM25]
    Q --> VEC[密ベクトル]
    Q --> PATH[PathRAG パス]
    Q --> GS[コミュニティ要約]
    PPR & BM & VEC --> RRF[RRF 統合]
    ACL{{根拠ベース可視性 / RLS<br/>as_of / 露出バジェット}} -.-> Q
  end
  Q -- "引用付きコンテキスト" --> A
```

- **テーブル**: `graph`, `document`(uuidv7), `chunk`(ACL, lexemes, embedding), `entity`(canonical_key, aliases, key_grams), `mention`(エンティティ×チャンクと、そのチャンクでの記述・別名・埋め込み・metadata), `relation`(valid_during, WITHOUT OVERLAPS), `relation_evidence`, `community`, `community_member`, `report`(LLM 生成文、required_acls, stale, generated_by), `exposure`, `erasure_log`
- **ビュー**: `fact`（人が読める事実一覧）、`pending_report`（要約が必要な作業キュー）
- **トリガ**: チャンクの ACL 継承と字句索引、エンティティのバイグラム索引、根拠の追加・削除による要約の `stale` 化（遷移テーブルを使う文単位トリガ）、文書とチャンクの ACL の変更を `required_acls` に伝播
- **Rust モジュール**: `text.rs`（日本語処理・BM25・チャンク分割）、`ppr.rs`、`leiden.rs`、`paths.rs`、`vector.rs` は PostgreSQL に依存しない純 Rust で単体テストできます。`api/` が SPI を介した SQL 関数です

## 5. ビルドとインストール

必要なもの: PostgreSQL 18（サーバ開発ヘッダを含む）、Rust 1.96 以上、`cargo-pgrx` 0.19.3、libclang

```bash
cargo install cargo-pgrx --version 0.19.3 --locked
cargo pgrx init --pg18 $(which pg_config)      # PG18 の pg_config を指定
cargo pgrx install --release --pg-config $(which pg_config)
```

```sql
CREATE EXTENSION pg_graphrag CASCADE;   -- btree_gist も入ります（WITHOUT OVERLAPS 用）
```

テスト:

```bash
cargo pgrx test pg18
# root で実行する場合（PostgreSQL は root では起動できないため）
cargo pgrx test pg18 --runas postgres --pgdata /var/tmp/pgrx-test
```

## 6. 使い方

一通りの流れは [`examples/quickstart.sql`](examples/quickstart.sql)（API キー不要、LLM 出力は JSON で同梱）にあります。

```sql
-- 1) グラフと文書（チャンク分割は自動。acl は読めるロール）
SELECT graphrag.create_graph('demo');
SELECT graphrag.add_document('demo', content => '…', external_id => 'd4', acl => ARRAY['hr']);

-- 2) LLM の抽出結果をチャンクごとに保存（GraphRAG / LightRAG 形式の JSON をそのまま受け付けます）
SELECT graphrag.ingest_extraction(chunk_id, '{"entities":[…], "relations":[…]}');
--    あるいは LLM を使わずに索引を作る
SELECT graphrag.lazy_index('demo');

-- 3) 検索
SELECT * FROM graphrag.local_search('demo', query => '…', query_embedding => $1, top_k => 10);
SELECT * FROM graphrag.relational_paths('demo', ARRAY['山田花子', '神奈川県']);
SELECT graphrag.build_context('demo', query => '…', max_tokens => 4000);

-- 4) コミュニティ要約（変わったものだけ LLM に送る）
SELECT graphrag.build_communities('demo');
SELECT ref_id, graphrag.community_context(ref_id) FROM graphrag.pending_report WHERE kind = 'community';
SELECT graphrag.set_report('community', 42, '要約…', title => '…', generated_by => 'claude-opus-5-5');
SELECT * FROM graphrag.global_search('demo', query => 'このコーパスの主なテーマは？');

-- 5) 時間・削除
SELECT graphrag.upsert_relation('demo', 'ミナト製作所', '鈴木一郎', 'led_by', valid_from => '2026-04-01', exclusive => true);
SELECT * FROM graphrag.local_search('demo', query => '…', as_of => '2023-01-01');
SELECT graphrag.forget_document('demo', 'd4');
```

LLM（Claude）と組み合わせたパイプラインの例は [`examples/graphrag_pipeline.py`](examples/graphrag_pipeline.py) です。構造化出力で抽出し、`pending_report` に出たコミュニティだけを要約し、エンドユーザーのロールで `build_context` を呼んで出典付きで回答します。

```bash
pip install -r examples/requirements.txt
export DATABASE_URL=postgresql:///demo
python examples/graphrag_pipeline.py ingest  --graph docs ./texts/*.txt
python examples/graphrag_pipeline.py reports --graph docs
python examples/graphrag_pipeline.py ask     --graph docs --role analyst "質問"
python examples/graphrag_pipeline.py lazy    --graph docs ./texts/*.txt          # LLM トークン 0
python examples/graphrag_pipeline.py ask     --graph docs --context-only "質問"   # LLM を呼ばずにコンテキストだけ表示
```

## 7. API リファレンス

すべて `graphrag` スキーマにあります。`as_of` を省略すると `now()` 時点の事実を使います。`options jsonb` で個別の調整ができます（後述）。

### 取り込み

| 関数 | 説明 |
|---|---|
| `create_graph(name, config)` → int | グラフを作成します（冪等） |
| `drop_graph(name)` → bool | グラフと中身をすべて削除します |
| `add_document(graph, content, external_id, title, metadata, acl, source_uri, chunk_tokens=400, chunk_overlap=40)` → bigint | 文書を登録し、文境界を考慮してチャンクに分割します |
| `add_chunk(document_id, content, ord, embedding)` → bigint | 自前で分割したチャンクを追加します |
| `upsert_entity(graph, name, entity_type, description, chunk_id, embedding, aliases, metadata)` → bigint | エンティティを登録または統合します。`chunk_id` があれば出典付きで、別名・説明・埋め込み・metadata は言及に保存します（そのチャンクを読めるロールにだけ見えます）。なければキュレーション済み（全員に可視）として、エンティティ自体に保存します |
| `upsert_relation(graph, source, target, rel_type, description, chunk_id, weight, valid_from, valid_to, exclusive, source_type, target_type)` → bigint | 関係を登録または補強します（端点がなければ作成）。`exclusive` で同種の現行事実を閉じます。ただし、新しい根拠を読めないロールにも見えている事実は閉じず、NOTICE を出します |
| `ingest_extraction(chunk_id, jsonb)` → jsonb | LLM の抽出結果をまとめて保存します。`name/entity_name`、`source/src_id`、`type/relation/relationship`、`weight/strength`、`valid_from/valid_at` などの表記ゆれを受け付けます |
| `lazy_index(graph, document_id, max_phrases=12)` → jsonb | LLM なしでキーフレーズ共起グラフを作ります |

### 名寄せ

| 関数 | 説明 |
|---|---|
| `canonical_key(text)` | 正規化キー（`トヨタ自動車株式会社` / `トヨタ自動車(株)` / `トヨタ自動車㈱` → `トヨタ自動車`、`ＩＢＭ` / `IBM Corp.` → `ibm`） |
| `name_similarity(a, b)` | Jaro–Winkler と文字バイグラム Dice の平均（CJK でも有効） |
| `match_entities(graph, names[], min_score=0.6, per_name=3)` | GIN バイグラム索引を使うあいまい検索 |
| `find_duplicates(graph, threshold=0.85, embedding_weight=0.3, max_pairs=100)` | 名前と埋め込みの類似度で重複候補を出します（型が違えば減点）。`grant_writer` を付けたロールだけが実行できます |
| `entity_aliases(entity_id)` / `entity_embedding(entity_id)` | 呼び出し側に見える別名と埋め込み（キュレーション済みのものと、見える言及に記録されたもの） |
| `merge_entities(keep_id, merge_ids[])` → jsonb | 言及・関係（有効期間と根拠を含む）・別名・埋め込みを統合し、自己ループを除きます |

### 検索

| 関数 | 説明 |
|---|---|
| `local_search(graph, query, query_embedding, seed_entities[], top_k=10, as_of, options)` | PPR・密ベクトル・BM25 を RRF で統合したチャンク検索。`graph_score / dense_score / lexical_score` と関連エンティティを返します |
| `rank_entities(...)` | PPR によるエンティティの順位 |
| `lexical_search(graph, query, top_k)` | CJK バイグラム BM25 |
| `global_search(graph, query, query_embedding, level, top_k)` | コミュニティ要約の検索。`visibility` は `full` / `redacted` / `no_report` のいずれかです |
| `relational_paths(graph, entities[], max_hops=3, top_k=10, as_of, options)` | 説明可能な関係パス（信頼度、エンティティ ID、関係 ID、根拠チャンク ID） |
| `subgraph(graph, entities[], hops=1, max_nodes=50, as_of)` → jsonb | 可視化用の近傍グラフ |
| `build_context(graph, query, query_embedding, seed_entities[], max_tokens=3000, as_of, options)` → jsonb | 出典付きでトークン予算内のコンテキスト（`context`, `sections`, `seeds`, `ppr`, `token_estimate`） |

`options` の主なキー: `damping`, `epsilon`, `max_nodes`, `seed_ids`（pgvector など外部検索で得たエンティティ ID）, `embedding_seed_k`, `max_seeds`, `ppr_entities`, `candidate_k`, `graph_weight` / `dense_weight` / `lexical_weight`, `rrf_k`, `lexical`(bool), `paths`(bool), `max_hops`, `decay`, `threshold`, `chunks`, `entities`, `entity_share` / `relation_share` / `path_share` / `community_share`

### コミュニティと要約

| 関数 | 説明 |
|---|---|
| `build_communities(graph, resolution=1.0, max_cluster_size=10, max_levels=3, seed=42, min_size=2)` → jsonb | 階層 Leiden を実行し、変わっていないコミュニティの要約を引き継ぎます |
| `community_context(community_id, max_tokens=6000)` | LLM に渡す要約用の入力（下位コミュニティの要約、メンバー、内部関係）。`grant_writer` を付けたロールだけが実行できます |
| `set_report(kind, ref_id, content, title, embedding, generated_by)` | `entity` / `relation` / `community` の要約を保存し、`required_acls` を自動計算します |
| ビュー `pending_report` | 要約がない、または古くなったものの一覧 |

### 統治

| 関数 | 説明 |
|---|---|
| `forget_document(document_id, clear_embeddings=true)` / `forget_document(graph, external_id, …)` → jsonb | 文書と、そこから派生した知識を削除します。呼び出し側には文書自体を消す権限（`graphrag.document` の DELETE と、RLS が有効なら文書の ACL）が必要です。派生物は、呼び出し側に見えないものも含めて削除します |
| `invalidate_relation(relation_id, at)` | 事実の有効期間を閉じます（削除はしません） |
| `enable_row_security()` | 直接の SQL アクセスにも根拠ベースの RLS をかけます |
| `grant_reader(role)` / `grant_writer(role)` | アプリケーション用ロールに権限を付与します。RLS が無効なときは、テーブルを直接読めてしまうことを WARNING で知らせます |
| `exposure_report(graph)` / `reset_exposure(graph, principal)` | ロールごとの開示状況 |
| `stats(graph)` → jsonb | 件数・古くなった要約・削除履歴など |

### ユーティリティ

`lexical_tokens(text)`, `estimate_tokens(text)`, `extract_keyphrases(text, max)`, `chunk_text(text, max_tokens, overlap_tokens)`, `cosine_similarity(real[], real[])`, `acl_is_public(text[])`, `acl_visible(text[])`, `acls_satisfied(jsonb)`

## 8. セキュリティモデル

- **ACL ラベルは PostgreSQL のロール名です**。`acl IS NULL` または空配列は公開、`'public'` を含む場合も公開です。所属判定は `pg_has_role(…, 'USAGE')` で行うため、ネストしたロールも DB 自身が判断します。PostgreSQL が権限を適用するときと同じく、`WITH INHERIT FALSE` の所属（CREATEROLE を持つロールが、自分の作ったロールに ADMIN だけで所属する場合など）は数えません。superuser はすべてを見られます。
- **可視性の導出**: チャンク → （そのチャンクを根拠とする）言及・関係 → エンティティ・関係 → グラフ探索、の順に伝わります。エンティティと関係は、見える根拠が 1 つ以上あるか、キュレーション済み（チャンクなしで登録）の場合だけ見えます。PPR とパス探索は、見えない辺や見えない端点を一切たどりません。
- **抽出結果の属性は根拠の側に置きます**: チャンクから抽出した別名・説明・埋め込み・metadata は言及（`mention`）に保存し、そのチャンクを読めるロールにだけ見せます（`entity_aliases()` / `entity_embedding()`）。エンティティの行に残るのは、最初に登録された名前、型ラベル、キュレーション済みの属性だけです。そのため、公開の根拠を 1 つでも持つエンティティでも、制限付き文書の別名や埋め込みは他のロールに見えません。チャンクや文書の ACL を後から変えても、この規則はそのまま保たれます。
- **排他的な事実の更新**: `exclusive` の関係が現行の事実を閉じるのは、閉じられる事実を見られるロール全員が、新しい根拠も読める場合だけです。そうでなければ、制限付き文書が他のロールに見える事実の有効期間を変え、いつ変わったかが漏れてしまいます。
- **スコアの統計**: BM25 の文書数・文書頻度と、シードのノード特異性は、呼び出し側に見えるチャンクだけで数えます。見えない文書を数えると、スコアの変化から秘匿文書に含まれる語を推測できるためです。
- **派生テキスト**（要約）は、生成に使った全根拠の ACL 集合 `required_acls` をすべて満たす場合だけ表示します。満たさない場合、エンティティの説明は見えるチャンクの記述だけから組み立て直し、コミュニティは伏せ字（`redacted`）にして見えるメンバーだけを返します。`required_acls` の計算は SECURITY DEFINER 関数で行うため、呼び出し側に RLS がかかっていても漏れなく集計されます。文書 ACL とチャンク ACL のどちらを変えても、影響する要約の `required_acls` を計算し直し、`stale` にします。
- **アプリケーションからの使い方**: 接続プールを使う場合は、リクエストごとに `SET LOCAL ROLE <end_user_role>` を実行してから検索関数を呼んでください。`enable_row_security()` を実行すると、テーブルを直接 `SELECT` しても同じ規則が適用されます（書き込みは `GRANT` で管理します）。RLS が有効な場合、閲覧者は存在自体が見えない要約を `no_report` として受け取ります。RLS は表の所有者以外の全ロールにかかるため、自分が所属しない ACL の文書を取り込む書き込み用ロールには `BYPASSRLS` が必要です（または表の所有者で実行します）。
- **露出バジェット**: `exposure_tracking` と `exposure_budget` は `SUSET` パラメータなので、一般ロールは無効化できません。記録は SECURITY DEFINER 関数を通して行い、`session_user` が対象ロールになれること（`SET` 権限）を検証します。その関数は、対象ロールに見えない ID や存在しない ID を無視するため、隠れたエンティティの存在確認には使えません。`local_search`・`rank_entities`・`relational_paths`・`subgraph`・`global_search`・`match_entities`・`lexical_search`・`build_context`（コミュニティ要約とシードを含む）が開示を記録します。グラフ全体を読む保守用の関数（`community_context`・`find_duplicates`）は、`grant_writer` を付けたロールだけが実行できます。

## 9. 設定パラメータ

| GUC | 既定値 | 説明 |
|---|---|---|
| `graphrag.ppr_damping` | 0.5 | 辺をたどる確率（HippoRAG と同じ値） |
| `graphrag.ppr_epsilon` | 1e-4 | ローカルプッシュの残差閾値。小さいほど精度が上がり、読むノードが増えます |
| `graphrag.max_expansion` | 20000 | 1 回の検索で隣接リストを読むノード数の上限 |
| `graphrag.exposure_tracking` | off | ロールごとの開示記録（superuser のみ変更可） |
| `graphrag.exposure_budget` | 0 | ロールに開示してよいエンティティの割合の上限（0 は無制限、superuser のみ変更可） |

## 10. テスト

`cargo pgrx test pg18` で 44 件のテストを実行し、すべて成功しています（PostgreSQL 18.6 で確認）。

- **純 Rust の単体テスト（21 件）**: 正規化キー（全角・半角、法人格、中点）、類似度、CJK バイグラム、キーフレーズ（日英、`AI倫理` のような英字と漢字の複合語）、トークン推定、チャンク分割、BM25、ローカルプッシュ PPR と厳密解（べき乗法）の一致、局所性、ノード上限、Leiden（2 つのクリーク、Zachary の空手クラブでモジュラリティ 0.40 超、全コミュニティの連結性、決定性、階層の整合性）、PathRAG のハブ枝刈りと向き
- **DB 内の結合テスト（23 件）**: 名寄せと出典、PPR による多段検索（グラフ距離の順になること）、説明可能なパス、日本語 BM25、権限（RLS なしとあり、SET ROLE による 2 ロール比較、要約の伏せ字化）、時間（排他的な事実の無効化、`as_of`、WITHOUT OVERLAPS 違反の検出）、削除（派生知識と要約の削除、監査記録）、コミュニティ（要約の引き継ぎ、根拠追加による stale 化）、伏せ字化されたグローバル検索、重複検出と統合、LLM なし索引、密ベクトル検索、文書 ACL 変更の要約への伝播、コンテキストの出典と予算、露出バジェット。セキュリティの回帰テストとして、制限付き抽出が共有エンティティ・事実に漏れないこと、チャンク ACL 変更の要約への伝播、RLS 下での削除の完全性と権限、BM25 統計が見えない文書に影響されないこと、`INHERIT FALSE` の所属では読めないこと、露出記録関数で隠れたエンティティを探れないこと、保守用関数を閲覧ロールが呼べないこと

## 11. 制約と今後の課題

- **ベクトル検索は全件走査です**（`real[]` で保持し、pgvector に依存しないため）。大規模なコーパスでは、pgvector の HNSW 索引（例: `CREATE INDEX ON graphrag.entity USING hnsw ((embedding::vector(1024)) vector_cosine_ops)`）で上位のエンティティを求め、`options => '{"seed_ids": [...]}'` で渡してください。pgvector の型を任意依存として直接扱う対応は今後の課題です。
- **Leiden の細分化段階は貪欲法です**（論文の θ → 0 の極限）。連結性の保証は保たれますが、ランダム化版より探索の幅は狭くなります。
- **`lazy_index` のキーフレーズ抽出は文字種によるヒューリスティック**で、形態素解析は行いません。精度が必要な場合は、外部の形態素解析器や LLM の抽出結果を `ingest_extraction` に渡してください。
- **トークン数は推定値**です（CJK は 1 文字 1 トークン、それ以外は 4 文字 1 トークン）。
- **露出バジェットは開示済みエンティティの割合で数えます**。意味的に同等な言い換えや、複数ロールの結託は検出しません。
- **エンティティの名前と型ラベル、関係の重みは根拠ごとに分けていません**。名前は最初に登録した表記、型は最初に付いたラベルで、関係の重みは見えない根拠の分も含めて加算されます。
- **権限ごとに分けた要約**（ACL 区画ごとのコミュニティ要約）は未実装です。現状は、全根拠を満たすロールにだけ要約を見せる保守的な方式です。
- このリポジトリでの対象は PostgreSQL 18 です（`uuidv7()` と `WITHOUT OVERLAPS` を使用）。

## 12. 参考文献

- Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* 2024. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
- Gutiérrez et al. *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* NeurIPS 2024. [arXiv:2405.14831](https://arxiv.org/abs/2405.14831)
- Gutiérrez et al. *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models* (HippoRAG 2). 2025. [arXiv:2502.14802](https://arxiv.org/abs/2502.14802)
- Guo et al. *LightRAG: Simple and Fast Retrieval-Augmented Generation.* 2024. [arXiv:2410.05779](https://arxiv.org/abs/2410.05779)
- Chen et al. *PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths.* 2025. [arXiv:2502.14902](https://arxiv.org/abs/2502.14902)
- *LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora.* 2025. [arXiv:2510.10114](https://arxiv.org/abs/2510.10114)
- *Efficient Retrieval-Augmented Generation via Token Co-occurrence Graphs* (TIGRAG). 2026. [arXiv:2606.30093](https://arxiv.org/abs/2606.30093)
- Rasmussen et al. *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.* 2025. [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)
- Traag, Waltman, van Eck. *From Louvain to Leiden: guaranteeing well-connected communities.* Scientific Reports 9, 5233 (2019). [doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z)
- Andersen, Chung, Lang. *Local Graph Partitioning using PageRank Vectors.* FOCS 2006.
- Cormack, Clarke, Büttcher. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.
- Liu, Zhang, Wang. *Exposing Privacy Risks in Graph Retrieval-Augmented Generation.* Findings of ACL 2026. [arXiv:2508.17222](https://arxiv.org/abs/2508.17222)
- *GraphSteal: Structural Knowledge Stealing from Graph RAG via Traversal Reconstruction.* 2026. [arXiv:2605.28645](https://arxiv.org/abs/2605.28645)
- Peng et al. *Graph Retrieval-Augmented Generation: A Survey.* 2024. [arXiv:2408.08921](https://arxiv.org/abs/2408.08921)
- Han et al. *Retrieval-Augmented Generation with Graphs (GraphRAG).* 2025. [arXiv:2501.00309](https://arxiv.org/abs/2501.00309)
- *In-depth Analysis of Graph-based RAG in a Unified Framework.* 2025. [arXiv:2503.04338](https://arxiv.org/abs/2503.04338)
- PostgreSQL 18 Release Notes. <https://www.postgresql.org/docs/18/release-18.html>

