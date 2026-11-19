const readOperations = new Set(["events", "snapshot"]);
const writeOperations = new Set(["probe", "admit", "simulate", "release", "alarm", "failover"]);

type Context = { params: Promise<{ operation: string }> };

async function forward(request: Request, { params }: Context) {
  const { operation } = await params;
  const post = request.method === "POST";
  if (!(post ? writeOperations : readOperations).has(operation)) {
    return Response.json({ detail: "Unknown operation." }, { status: 404 });
  }
  const contentType = request.headers.get("content-type");
  if (post && (request.headers.get("sec-fetch-site") === "cross-site" ||
      contentType?.split(";")[0].trim().toLowerCase() !== "application/json")) {
    return Response.json({ detail: "A same-site JSON request is required." }, { status: 415 });
  }

  const backend = process.env.NURSE_API_URL || "http://127.0.0.1:9501";
  try {
    const upstream = await fetch(new URL(operation, backend), {
      method: request.method,
      headers: post ? { "Content-Type": "application/json" } : undefined,
      body: post ? await request.text() : undefined,
      cache: "no-store",
      signal: request.signal,
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
        "Cache-Control": "no-store, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return Response.json({ detail: "The Nurse Station service is unavailable." }, { status: 503 });
  }
}

export const GET = forward;
export const POST = forward;
