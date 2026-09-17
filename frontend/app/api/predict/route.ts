import { NextRequest, NextResponse } from "next/server";

// Server-side only — never exposed to the browser. This is the address of
// the internal FastAPI service; the browser only ever talks to this Next.js
// route, at /api/predict, on the same origin as the rest of the app.
const INTERNAL_API_URL = process.env.INTERNAL_API_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Forward the same multipart form data to the FastAPI backend.
    const forwardFormData = new FormData();
    forwardFormData.append("file", file);

    const backendRes = await fetch(`${INTERNAL_API_URL}/predict`, {
      method: "POST",
      body: forwardFormData,
    });

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: `Backend returned ${backendRes.status}` },
        { status: 502 }
      );
    }

    const result = await backendRes.json();
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Unknown error contacting the classifier" },
      { status: 500 }
    );
  }
}
