import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { loadConfig } from "./config.js";
import { createKnowledgeMcpServer } from "./server.js";

const config = loadConfig();
const app = express();

app.use(express.json({ limit: "2mb" }));

app.get("/health", async (_req, res) => {
  const qdrant = Object.fromEntries(await Promise.all(
    [
      ["ifsqn", config.IFSQN_QDRANT_URL],
      ["elsmar", config.ELSMAR_QDRANT_URL],
    ].map(async ([name, url]) => {
      try {
        const response = await fetch(new URL("/collections", url));
        return [name, response.ok ? "ok" : "error"];
      } catch {
        return [name, "error"];
      }
    }),
  ));

  res.json({
    status: "ok",
    transport: "streamable-http",
    path: config.mcpPath,
    publicBaseUrl: config.publicBaseUrl,
    collections: {
      ifsqn: config.IFSQN_COLLECTION,
      elsmar: config.ELSMAR_COLLECTION,
    },
    embeddingModel: config.OPENAI_EMBEDDING_MODEL,
    qdrant,
  });
});

for (const method of ["get", "delete"] as const) {
  app[method](config.mcpPath, (_req, res) => {
    res.status(405).set("Allow", "POST").send("Method Not Allowed");
  });
}

app.post(config.mcpPath, async (req, res) => {
  const server = createKnowledgeMcpServer(config);
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
  } as unknown as ConstructorParameters<typeof StreamableHTTPServerTransport>[0]);

  try {
    await server.connect(transport as never);
    await transport.handleRequest(req, res, req.body);

    res.on("close", () => {
      void transport.close();
      void server.close();
    });
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    process.stderr.write(`ifsqn-knowledge-mcp request failed: ${message}\n`);

    await transport.close().catch(() => undefined);
    await server.close().catch(() => undefined);

    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32603,
          message: "Internal server error",
        },
        id: null,
      });
    }
  }
});

const listener = app.listen(config.PORT, config.HOST, (error?: Error) => {
  if (error) {
    process.stderr.write(`ifsqn-knowledge-mcp failed to start: ${error.stack ?? error.message}\n`);
    process.exit(1);
  }

  process.stderr.write(
    `ifsqn-knowledge-mcp listening on http://${config.HOST}:${config.PORT}${config.mcpPath}\n`,
  );
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    listener.close(() => process.exit(0));
  });
}
