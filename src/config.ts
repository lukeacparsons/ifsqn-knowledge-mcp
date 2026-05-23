import { z } from "zod";

const ConfigSchema = z.object({
  HOST: z.string().trim().min(1).default("127.0.0.1"),
  PORT: z.coerce.number().int().min(1).max(65535).default(3200),
  MCP_PATH: z.string().trim().min(1).default("/mcp"),
  MCP_PUBLIC_BASE_URL: z.string().url().optional(),
  OPENAI_API_KEY: z.string().trim().min(1),
  OPENAI_EMBEDDING_MODEL: z.string().trim().min(1).default("text-embedding-3-large"),
  IFSQN_QDRANT_URL: z.string().url(),
  IFSQN_COLLECTION: z.string().trim().min(1).default("ifsqn_forum_posts_large"),
  ELSMAR_QDRANT_URL: z.string().url(),
  ELSMAR_COLLECTION: z.string().trim().min(1).default("elsmar_forum_posts_large"),
  MAX_CANDIDATES: z.coerce.number().int().min(5).max(200).default(60),
  MAX_RESULTS: z.coerce.number().int().min(1).max(50).default(20),
  MCP_BEARER_TOKEN: z.string().trim().min(8).optional(),
});

export type AppConfig = z.infer<typeof ConfigSchema> & {
  mcpPath: string;
  publicBaseUrl: string;
};

function normalizePath(path: string): string {
  return path.startsWith("/") ? path : `/${path}`;
}

export function loadConfig(env: Record<string, string | undefined> = process.env): AppConfig {
  const parsed = ConfigSchema.parse(env);
  const mcpPath = normalizePath(parsed.MCP_PATH);

  return {
    ...parsed,
    mcpPath,
    publicBaseUrl: parsed.MCP_PUBLIC_BASE_URL ?? `http://${parsed.HOST}:${parsed.PORT}${mcpPath}`,
  };
}
