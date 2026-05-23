import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { AppConfig } from "./config.js";
import { searchKnowledgeBase } from "./search.js";

const SearchFiltersSchema = z.object({
  topic_id: z.number().int().optional(),
  forum_id: z.number().int().optional(),
  author_name: z.string().min(1).optional(),
  category_title: z.string().min(1).optional(),
  forum_title: z.string().min(1).optional(),
}).optional();

const SearchInputSchema = {
  query: z.string().min(1).describe("Natural-language search query."),
  top_k: z.number().int().min(1).max(20).optional().describe("Number of final chunks to return."),
  candidate_k: z.number().int().min(5).max(100).optional().describe("Number of vector candidates to retrieve before reranking."),
  filters: SearchFiltersSchema.describe("Optional exact metadata filters."),
};

const readOnlyAnnotations = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
} as const;

function jsonContent(value: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }],
    structuredContent: value as Record<string, unknown>,
  };
}

function errorContent(error: unknown) {
  return {
    content: [{
      type: "text" as const,
      text: error instanceof Error ? error.message : String(error),
    }],
    isError: true as const,
  };
}

export function createKnowledgeMcpServer(config: AppConfig): McpServer {
  const server = new McpServer({
    name: "ifsqn-knowledge-mcp",
    version: "0.1.0",
  });

  server.registerTool(
    "search_knowledge_base",
    {
      title: "Search IFSQN knowledge base",
      description: "Search the embedded IFSQN forum corpus and return citation-ready chunks with source URLs and metadata.",
      inputSchema: SearchInputSchema,
      annotations: readOnlyAnnotations,
    },
    async (input) => {
      try {
        return jsonContent(await searchKnowledgeBase(config, input));
      } catch (error) {
        return errorContent(error);
      }
    },
  );

  return server;
}
