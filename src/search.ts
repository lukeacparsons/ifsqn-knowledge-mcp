import type { AppConfig } from "./config.js";

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

export type SearchResult = {
  rank: number;
  corpus: "ifsqn" | "elsmar";
  score: number;
  vector_score: number;
  lexical_score: number;
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
    retrieved_candidates: number;
  }>;
  embedding_model: string;
  retrieved_candidates: number;
  results: SearchResult[];
};

type CorpusConfig = {
  corpus: "ifsqn" | "elsmar";
  qdrantUrl: string;
  collection: string;
};

function asString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function asScalar(value: unknown): string | number | null {
  return typeof value === "string" || typeof value === "number" ? value : null;
}

function tokenize(value: string): string[] {
  return Array.from(new Set(
    value
      .toLowerCase()
      .split(/[^a-z0-9]+/i)
      .map((token) => token.trim())
      .filter((token) => token.length >= 3),
  ));
}

function lexicalScore(payload: Record<string, unknown>, query: string): number {
  const tokens = tokenize(query);
  if (tokens.length === 0) {
    return 0;
  }

  const haystack = [
    payload.topic_title,
    payload.forum_title,
    payload.category_title,
    payload.author_name,
    payload.text,
  ].filter((value): value is string => typeof value === "string").join("\n").toLowerCase();

  const hits = tokens.reduce((count, token) => count + (haystack.includes(token) ? 1 : 0), 0);
  return hits / tokens.length;
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

function toResult(corpus: "ifsqn" | "elsmar", point: QdrantPoint, query: string, rank: number): SearchResult {
  const payload = point.payload ?? {};
  const text = asString(payload.text) ?? "";
  const lexical = lexicalScore(payload, query);
  const combined = (point.score * 0.82) + (lexical * 0.18);
  const topicTitle = asString(payload.topic_title) ?? asString(payload.post_title);
  const sourceUrl = asString(payload.post_url) ?? asString(payload.source_url) ?? asString(payload.topic_url);

  return {
    rank,
    corpus,
    score: Number(combined.toFixed(6)),
    vector_score: Number(point.score.toFixed(6)),
    lexical_score: Number(lexical.toFixed(6)),
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
    text,
    citation: {
      title: topicTitle,
      url: sourceUrl,
      post_id: asScalar(payload.post_id),
      topic_id: asScalar(payload.topic_id),
    },
  };
}

function getCorpora(config: AppConfig, requested: SearchInput["corpus"]): CorpusConfig[] {
  const corpus = requested ?? "both";
  const configs: Record<"ifsqn" | "elsmar", CorpusConfig> = {
    ifsqn: {
      corpus: "ifsqn",
      qdrantUrl: config.IFSQN_QDRANT_URL,
      collection: config.IFSQN_COLLECTION,
    },
    elsmar: {
      corpus: "elsmar",
      qdrantUrl: config.ELSMAR_QDRANT_URL,
      collection: config.ELSMAR_COLLECTION,
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
    corpora.map(async (corpus) => ({
      corpus,
      points: await searchQdrant(config, corpus, vector, input),
    })),
  );
  const results = searches
    .flatMap(({ corpus, points }) => points.map((point, index) => toResult(corpus.corpus, point, input.query, index + 1)))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((result, index) => ({ ...result, rank: index + 1 }));
  const searchedCorpora = searches.map(({ corpus, points }) => ({
    corpus: corpus.corpus,
    collection: corpus.collection,
    retrieved_candidates: points.length,
  }));

  return {
    query: input.query,
    corpus: requestedCorpus,
    searched_corpora: searchedCorpora,
    embedding_model: config.OPENAI_EMBEDDING_MODEL,
    retrieved_candidates: searchedCorpora.reduce((sum, item) => sum + item.retrieved_candidates, 0),
    results,
  };
}
