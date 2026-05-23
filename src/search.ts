import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { AppConfig } from "./config.js";

const execFileAsync = promisify(execFile);

export type SearchFilters = {
  topic_id?: number | undefined;
  forum_id?: number | undefined;
  author_name?: string | undefined;
  category_title?: string | undefined;
  forum_title?: string | undefined;
};

export type SearchInput = {
  query: string;
  corpus?: "ifsqn" | "elsmar" | "both" | undefined;
  top_k?: number | undefined;
  candidate_k?: number | undefined;
  filters?: SearchFilters | undefined;
};

type QdrantPoint = {
  id: string | number;
  score: number;
  payload?: Record<string, unknown> | undefined;
};

type KeywordRow = {
  post_id: number | string | null;
  topic_id: number | string | null;
  forum_id: number | string | null;
  topic_title: string | null;
  forum_title: string | null;
  category_title: string | null;
  author_name: string | null;
  posted_at: string | null;
  source_url: string | null;
  post_url: string | null;
  topic_url: string | null;
  is_best_answer: number | null;
  is_topic_starter: number | null;
  keyword_rank: number;
  snippet: string | null;
  text: string;
};

export type SearchResult = {
  rank: number;
  corpus: "ifsqn" | "elsmar";
  score: number;
  vector_score: number | null;
  keyword_score: number | null;
  lexical_score: number;
  matched_by: Array<"vector" | "keyword">;
  reason: string;
  id: string | number;
  post_id: number | string | null;
  topic_id: number | string | null;
  topic_title: string | null;
  forum_title: string | null;
  category_title: string | null;
  author_name: string | null;
  posted_at: string | null;
  source_url: string | null;
  post_url: string | null;
  chunk_index: number | string | null;
  snippet: string | null;
  text: string;
  citation: {
    title: string | null;
    url: string | null;
    post_id: number | string | null;
    topic_id: number | string | null;
  };
};

export type SearchOutput = {
  query: string;
  corpus: "ifsqn" | "elsmar" | "both";
  searched_corpora: Array<{
    corpus: "ifsqn" | "elsmar";
    collection: string;
    vector_candidates: number;
    keyword_candidates: number;
    fts_enabled: boolean;
  }>;
  embedding_model: string;
  retrieved_candidates: number;
  results: SearchResult[];
};

type CorpusConfig = {
  corpus: "ifsqn" | "elsmar";
  qdrantUrl: string;
  collection: string;
  ftsDbPath?: string | undefined;
};

type Candidate = {
  corpus: "ifsqn" | "elsmar";
  key: string;
  vector?: SearchResult | undefined;
  keyword?: SearchResult | undefined;
  vectorRank?: number | undefined;
  keywordRank?: number | undefined;
  keywordRawRank?: number | undefined;
};

function asString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function asScalar(value: unknown): string | number | null {
  return typeof value === "string" || typeof value === "number" ? value : null;
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function tokenize(value: string): string[] {
  return Array.from(new Set(
    value
      .toLowerCase()
      .split(/[^a-z0-9]+/i)
      .map((token) => token.trim())
      .filter((token) => token.length >= 2),
  ));
}

function lexicalScoreFromText(values: Array<string | null | undefined>, query: string): number {
  const tokens = tokenize(query);
  if (tokens.length === 0) {
    return 0;
  }

  const haystack = values.filter((value): value is string => typeof value === "string").join("\n").toLowerCase();
  const hits = tokens.reduce((count, token) => count + (haystack.includes(token) ? 1 : 0), 0);
  return hits / tokens.length;
}

function lexicalScore(payload: Record<string, unknown>, query: string): number {
  return lexicalScoreFromText([
    asString(payload.topic_title),
    asString(payload.forum_title),
    asString(payload.category_title),
    asString(payload.author_name),
    asString(payload.text),
  ], query);
}

function buildQdrantFilter(filters: SearchFilters | undefined) {
  if (!filters) {
    return undefined;
  }

  const must: Array<{ key: string; match: { value: string | number } }> = [];
  for (const [key, value] of Object.entries(filters)) {
    if (typeof value === "string" && value.trim().length > 0) {
      must.push({ key, match: { value: value.trim() } });
    } else if (typeof value === "number" && Number.isFinite(value)) {
      must.push({ key, match: { value } });
    }
  }

  return must.length > 0 ? { must } : undefined;
}

async function fetchJson(url: string, init: RequestInit): Promise<unknown> {
  const response = await fetch(url, init);
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    payload = await response.text();
  }

  if (!response.ok) {
    throw new Error(`Request failed ${response.status} ${response.statusText}: ${JSON.stringify(payload).slice(0, 800)}`);
  }

  return payload;
}

async function embedQuery(config: AppConfig, query: string): Promise<number[]> {
  const payload = await fetchJson("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      authorization: `Bearer ${config.OPENAI_API_KEY}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: config.OPENAI_EMBEDDING_MODEL,
      input: query,
    }),
  }) as { data?: Array<{ embedding?: number[] }> };

  const embedding = payload.data?.[0]?.embedding;
  if (!Array.isArray(embedding)) {
    throw new Error("OpenAI embedding response did not include an embedding.");
  }

  return embedding;
}

async function searchQdrant(config: AppConfig, corpus: CorpusConfig, vector: number[], input: SearchInput): Promise<QdrantPoint[]> {
  const candidateLimit = Math.min(
    input.candidate_k ?? config.MAX_CANDIDATES,
    config.MAX_CANDIDATES,
  );
  const url = new URL(`/collections/${encodeURIComponent(corpus.collection)}/points/search`, corpus.qdrantUrl);
  const payload = await fetchJson(url.toString(), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      vector,
      limit: candidateLimit,
      with_payload: true,
      with_vector: false,
      filter: buildQdrantFilter(input.filters),
    }),
  }) as { result?: QdrantPoint[] };

  if (!Array.isArray(payload.result)) {
    throw new Error("Qdrant search response did not include result points.");
  }

  return payload.result;
}

function sqlLiteral(value: string | number): string {
  if (typeof value === "number") {
    return String(value);
  }
  return `'${value.replace(/'/g, "''")}'`;
}

function ftsPhrase(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

function buildFtsMatchQuery(query: string): string {
  const tokens = tokenize(query)
    .filter((token) => !["and", "are", "for", "from", "has", "have", "the", "this", "that", "with", "you", "your"].includes(token));

  if (tokens.length === 0) {
    return ftsPhrase(query.trim());
  }

  return tokens.map(ftsPhrase).join(" OR ");
}

function buildKeywordWhere(matchQuery: string, filters: SearchFilters | undefined): string {
  const clauses = [`posts_fts MATCH ${sqlLiteral(matchQuery)}`];

  if (!filters) {
    return clauses.join(" AND ");
  }

  if (typeof filters.topic_id === "number") {
    clauses.push(`topic_id = ${sqlLiteral(filters.topic_id)}`);
  }
  if (typeof filters.forum_id === "number") {
    clauses.push(`forum_id = ${sqlLiteral(filters.forum_id)}`);
  }
  if (filters.author_name) {
    clauses.push(`author_name = ${sqlLiteral(filters.author_name)}`);
  }
  if (filters.category_title) {
    clauses.push(`category_title = ${sqlLiteral(filters.category_title)}`);
  }
  if (filters.forum_title) {
    clauses.push(`forum_title = ${sqlLiteral(filters.forum_title)}`);
  }

  return clauses.join(" AND ");
}

async function searchKeyword(config: AppConfig, corpus: CorpusConfig, input: SearchInput): Promise<KeywordRow[]> {
  if (!corpus.ftsDbPath) {
    return [];
  }

  const candidateLimit = Math.min(
    input.candidate_k ?? config.MAX_CANDIDATES,
    config.MAX_CANDIDATES,
  );
  const matchQuery = buildFtsMatchQuery(input.query);
  const where = buildKeywordWhere(matchQuery, input.filters);
  const sql = `
    SELECT
      post_id,
      topic_id,
      forum_id,
      topic_title,
      forum_title,
      category_title,
      author_name,
      posted_at,
      source_url,
      post_url,
      topic_url,
      is_best_answer,
      is_topic_starter,
      bm25(posts_fts) AS keyword_rank,
      snippet(posts_fts, 3, '[', ']', ' ... ', 36) AS snippet,
      text
    FROM posts_fts
    WHERE ${where}
    ORDER BY keyword_rank
    LIMIT ${candidateLimit}
  `;

  try {
    const { stdout } = await execFileAsync(config.SQLITE3_BIN, ["-readonly", "-json", corpus.ftsDbPath, sql], {
      maxBuffer: 50 * 1024 * 1024,
      timeout: 20_000,
    });
    if (stdout.trim() === "") {
      return [];
    }
    const parsed = JSON.parse(stdout) as Array<Record<string, unknown>>;
    return parsed.map((row) => ({
      post_id: asScalar(row.post_id),
      topic_id: asScalar(row.topic_id),
      forum_id: asScalar(row.forum_id),
      topic_title: asString(row.topic_title),
      forum_title: asString(row.forum_title),
      category_title: asString(row.category_title),
      author_name: asString(row.author_name),
      posted_at: asString(row.posted_at),
      source_url: asString(row.source_url),
      post_url: asString(row.post_url),
      topic_url: asString(row.topic_url),
      is_best_answer: asNumber(row.is_best_answer),
      is_topic_starter: asNumber(row.is_topic_starter),
      keyword_rank: asNumber(row.keyword_rank) ?? 0,
      snippet: asString(row.snippet),
      text: asString(row.text) ?? "",
    }));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`keyword search failed for ${corpus.corpus}: ${message}\n`);
    return [];
  }
}

function sourceUrlFromPayload(payload: Record<string, unknown>): string | null {
  return asString(payload.post_url) ?? asString(payload.source_url) ?? asString(payload.topic_url);
}

function resultKey(corpus: "ifsqn" | "elsmar", postId: number | string | null, fallback: string | number): string {
  return `${corpus}:${postId ?? fallback}`;
}

function vectorToResult(corpus: "ifsqn" | "elsmar", point: QdrantPoint, query: string): SearchResult {
  const payload = point.payload ?? {};
  const text = asString(payload.text) ?? "";
  const lexical = lexicalScore(payload, query);
  const topicTitle = asString(payload.topic_title) ?? asString(payload.post_title);
  const sourceUrl = sourceUrlFromPayload(payload);

  return {
    rank: 0,
    corpus,
    score: 0,
    vector_score: Number(point.score.toFixed(6)),
    keyword_score: null,
    lexical_score: Number(lexical.toFixed(6)),
    matched_by: ["vector"],
    reason: "Semantic vector match",
    id: point.id,
    post_id: asScalar(payload.post_id),
    topic_id: asScalar(payload.topic_id),
    topic_title: topicTitle,
    forum_title: asString(payload.forum_title),
    category_title: asString(payload.category_title),
    author_name: asString(payload.author_name),
    posted_at: asString(payload.posted_at),
    source_url: asString(payload.source_url),
    post_url: asString(payload.post_url),
    chunk_index: asScalar(payload.chunk_index),
    snippet: null,
    text,
    citation: {
      title: topicTitle,
      url: sourceUrl,
      post_id: asScalar(payload.post_id),
      topic_id: asScalar(payload.topic_id),
    },
  };
}

function keywordToResult(corpus: "ifsqn" | "elsmar", row: KeywordRow, query: string): SearchResult {
  const sourceUrl = row.post_url ?? row.source_url ?? row.topic_url;
  const lexical = lexicalScoreFromText([
    row.topic_title,
    row.forum_title,
    row.category_title,
    row.author_name,
    row.text,
  ], query);

  return {
    rank: 0,
    corpus,
    score: 0,
    vector_score: null,
    keyword_score: null,
    lexical_score: Number(lexical.toFixed(6)),
    matched_by: ["keyword"],
    reason: "Exact keyword/BM25 match",
    id: row.post_id ?? `${row.topic_id ?? "unknown"}:keyword`,
    post_id: row.post_id,
    topic_id: row.topic_id,
    topic_title: row.topic_title,
    forum_title: row.forum_title,
    category_title: row.category_title,
    author_name: row.author_name,
    posted_at: row.posted_at,
    source_url: row.source_url,
    post_url: row.post_url,
    chunk_index: null,
    snippet: row.snippet,
    text: row.text,
    citation: {
      title: row.topic_title,
      url: sourceUrl,
      post_id: row.post_id,
      topic_id: row.topic_id,
    },
  };
}

function mergeCandidates(
  corpus: "ifsqn" | "elsmar",
  vectorResults: SearchResult[],
  keywordResults: Array<{ result: SearchResult; rawRank: number }>,
): SearchResult[] {
  const candidates = new Map<string, Candidate>();

  vectorResults.forEach((result, index) => {
    const key = resultKey(corpus, result.post_id, result.id);
    candidates.set(key, {
      corpus,
      key,
      vector: result,
      vectorRank: index + 1,
    });
  });

  keywordResults.forEach(({ result, rawRank }, index) => {
    const key = resultKey(corpus, result.post_id, result.id);
    const existing = candidates.get(key);
    if (existing) {
      existing.keyword = result;
      existing.keywordRank = index + 1;
      existing.keywordRawRank = rawRank;
    } else {
      candidates.set(key, {
        corpus,
        key,
        keyword: result,
        keywordRank: index + 1,
        keywordRawRank: rawRank,
      });
    }
  });

  return Array.from(candidates.values()).map((candidate) => scoreCandidate(candidate));
}

function reciprocalRank(rank: number | undefined): number {
  return rank ? 1 / (60 + rank) : 0;
}

function scoreCandidate(candidate: Candidate): SearchResult {
  const preferred = candidate.vector ?? candidate.keyword;
  if (!preferred) {
    throw new Error("candidate has no result");
  }

  const keyword = candidate.keyword;
  const vector = candidate.vector;
  const matchedBy: Array<"vector" | "keyword"> = [];
  if (vector) {
    matchedBy.push("vector");
  }
  if (keyword) {
    matchedBy.push("keyword");
  }

  const vectorRrf = reciprocalRank(candidate.vectorRank);
  const keywordRrf = reciprocalRank(candidate.keywordRank);
  const lexical = Math.max(vector?.lexical_score ?? 0, keyword?.lexical_score ?? 0);
  const combined = (4.0 * vectorRrf) + (5.0 * keywordRrf) + (0.10 * lexical);
  const keywordScore = keyword ? Number((1 / Math.max(1, Math.abs(candidate.keywordRawRank ?? 1))).toFixed(6)) : null;
  const reason = matchedBy.length === 2
    ? "Matched by both semantic vector search and exact keyword/BM25 search"
    : matchedBy[0] === "keyword"
      ? "Exact keyword/BM25 match"
      : "Semantic vector match";

  return {
    ...preferred,
    score: Number(combined.toFixed(6)),
    vector_score: vector?.vector_score ?? null,
    keyword_score: keywordScore,
    lexical_score: Number(lexical.toFixed(6)),
    matched_by: matchedBy,
    reason,
    snippet: keyword?.snippet ?? vector?.snippet ?? null,
    text: vector?.text || keyword?.text || preferred.text,
    citation: {
      title: preferred.citation.title ?? keyword?.citation.title ?? vector?.citation.title ?? null,
      url: preferred.citation.url ?? keyword?.citation.url ?? vector?.citation.url ?? null,
      post_id: preferred.citation.post_id ?? keyword?.citation.post_id ?? vector?.citation.post_id ?? null,
      topic_id: preferred.citation.topic_id ?? keyword?.citation.topic_id ?? vector?.citation.topic_id ?? null,
    },
  };
}

function diversifyByTopic(results: SearchResult[], topK: number): SearchResult[] {
  const seenTopics = new Set<string>();
  const primary: SearchResult[] = [];
  const overflow: SearchResult[] = [];

  for (const result of results) {
    const topicKey = `${result.corpus}:${result.topic_id ?? result.post_id ?? result.id}`;
    if (seenTopics.has(topicKey)) {
      overflow.push(result);
    } else {
      seenTopics.add(topicKey);
      primary.push(result);
    }
  }

  return [...primary, ...overflow].slice(0, topK);
}

function getCorpora(config: AppConfig, requested: SearchInput["corpus"]): CorpusConfig[] {
  const corpus = requested ?? "both";
  const configs: Record<"ifsqn" | "elsmar", CorpusConfig> = {
    ifsqn: {
      corpus: "ifsqn",
      qdrantUrl: config.IFSQN_QDRANT_URL,
      collection: config.IFSQN_COLLECTION,
      ftsDbPath: config.IFSQN_FTS_DB_PATH,
    },
    elsmar: {
      corpus: "elsmar",
      qdrantUrl: config.ELSMAR_QDRANT_URL,
      collection: config.ELSMAR_COLLECTION,
      ftsDbPath: config.ELSMAR_FTS_DB_PATH,
    },
  };

  if (corpus === "both") {
    return [configs.ifsqn, configs.elsmar];
  }

  return [configs[corpus]];
}

export async function searchKnowledgeBase(config: AppConfig, input: SearchInput): Promise<SearchOutput> {
  const topK = Math.min(input.top_k ?? 8, config.MAX_RESULTS);
  const requestedCorpus = input.corpus ?? "both";
  const corpora = getCorpora(config, requestedCorpus);
  const vector = await embedQuery(config, input.query);
  const searches = await Promise.all(
    corpora.map(async (corpus) => {
      const [points, keywordRows] = await Promise.all([
        searchQdrant(config, corpus, vector, input),
        searchKeyword(config, corpus, input),
      ]);
      const vectorResults = points.map((point) => vectorToResult(corpus.corpus, point, input.query));
      const keywordResults = keywordRows.map((row) => ({
        result: keywordToResult(corpus.corpus, row, input.query),
        rawRank: row.keyword_rank,
      }));

      return {
        corpus,
        points,
        keywordRows,
        results: mergeCandidates(corpus.corpus, vectorResults, keywordResults),
      };
    }),
  );

  const rankedResults = searches
    .flatMap(({ results: corpusResults }) => corpusResults)
    .sort((a, b) => b.score - a.score);
  const results = diversifyByTopic(rankedResults, topK)
    .map((result, index) => ({ ...result, rank: index + 1 }));
  const searchedCorpora = searches.map(({ corpus, points, keywordRows }) => ({
    corpus: corpus.corpus,
    collection: corpus.collection,
    vector_candidates: points.length,
    keyword_candidates: keywordRows.length,
    fts_enabled: Boolean(corpus.ftsDbPath),
  }));

  return {
    query: input.query,
    corpus: requestedCorpus,
    searched_corpora: searchedCorpora,
    embedding_model: config.OPENAI_EMBEDDING_MODEL,
    retrieved_candidates: searchedCorpora.reduce((sum, item) => sum + item.vector_candidates + item.keyword_candidates, 0),
    results,
  };
}
