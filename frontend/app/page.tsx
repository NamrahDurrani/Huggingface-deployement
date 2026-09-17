"use client";

import { useState } from "react";

type PredictionResult = {
  predicted_class: string | null;
  class_scores: Record<string, number> | null;
  confidence: number | null;
  status: string;
  message?: string;
};

const CLASS_ORDER = ["NILM", "LSIL", "HSIL", "SCC"] as const;

// Severity tokens — NILM (normal) through SCC (most severe), matching how
// pathology reporting conventionally color-codes Bethesda-system categories.
const SEVERITY: Record<string, { bg: string; text: string; bar: string; label: string }> = {
  NILM: { bg: "bg-emerald-50", text: "text-emerald-700", bar: "bg-emerald-500", label: "Negative" },
  LSIL: { bg: "bg-amber-50", text: "text-amber-700", bar: "bg-amber-500", label: "Low-grade" },
  HSIL: { bg: "bg-orange-50", text: "text-orange-700", bar: "bg-orange-500", label: "High-grade" },
  SCC: { bg: "bg-rose-50", text: "text-rose-700", bar: "bg-rose-500", label: "Carcinoma" },
};

// Calls this Next.js app's own /api/predict route (same origin) — that
// route forwards to the internal FastAPI service server-side. The browser
// never talks to the Python backend directly.
const PREDICT_ENDPOINT = "/api/predict";

function ClinicalMark() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-[#2563a8]">
      <path
        d="M12 3v6M12 15v6M3 12h6M15 12h6"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <circle cx="12" cy="12" r="9.5" stroke="currentColor" strokeWidth="1.2" opacity="0.35" />
    </svg>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(PREDICT_ENDPOINT, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? `Could not reach the classifier: ${err.message}`
          : "Could not reach the classifier."
      );
    } finally {
      setLoading(false);
    }
  }

  const severity = result?.predicted_class ? SEVERITY[result.predicted_class] : null;

  return (
    <main className="min-h-screen bg-[#f6f8fa] text-[#1f2937]">
      <div className="mx-auto max-w-4xl px-6 py-14">
        {/* Header */}
        <header className="mb-8 flex items-start justify-between border-b border-[#e2e8f0] pb-6">
          <div className="flex items-center gap-3">
            <ClinicalMark />
            <div>
              <h1 className="text-lg font-semibold text-[#111827]">
                Cervical Cell Analysis
              </h1>
              <p className="text-sm text-[#6b7280]">
                Computational classification — NILM · LSIL · HSIL · SCC
              </p>
            </div>
          </div>
          <span className="rounded-full bg-[#eef2f7] px-3 py-1 text-xs font-medium text-[#4b5563]">
            v1.0 · research prototype
          </span>
        </header>

        {/* Main panel */}
        <section className="grid gap-6 sm:grid-cols-5">
          {/* Upload / preview */}
          <div className="sm:col-span-2">
            <div className="rounded-xl border border-[#e2e8f0] bg-white p-4 shadow-sm">
              <label
                htmlFor="file-upload"
                className="flex h-56 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-[#d1dbe5] bg-[#fafbfc] text-center transition-colors hover:border-[#2563a8]"
              >
                {previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={previewUrl}
                    alt="Selected cell image"
                    className="h-full w-full rounded-lg object-contain p-2"
                  />
                ) : (
                  <span className="px-6 text-sm text-[#8a94a3]">
                    Click to select an LBC cell image
                  </span>
                )}
              </label>
              <input
                id="file-upload"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />

              <button
                onClick={handleSubmit}
                disabled={!file || loading}
                className="mt-4 w-full rounded-lg bg-[#2563a8] py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#1e5289] disabled:cursor-not-allowed disabled:bg-[#c3cdd8]"
              >
                {loading ? "Analyzing…" : "Run classification"}
              </button>
            </div>
          </div>

          {/* Results */}
          <div className="sm:col-span-3">
            <div className="h-full rounded-xl border border-[#e2e8f0] bg-white p-6 shadow-sm">
              {!result && !error && (
                <div className="flex h-full flex-col items-center justify-center text-center text-sm text-[#9aa5b1]">
                  <p>Findings will appear here after analysis.</p>
                </div>
              )}

              {error && (
                <div className="rounded-lg bg-rose-50 p-4 text-sm text-rose-700">{error}</div>
              )}

              {result &&
                (result.status === "REJECTED_NO_CELL_DETECTED" ||
                  result.status === "REJECTED_NOT_CYTOLOGY_IMAGE") && (
                  <div className="rounded-lg bg-amber-50 p-4">
                    <p className="text-sm font-medium text-amber-800">Image rejected</p>
                    <p className="mt-1 text-sm text-amber-700">{result.message}</p>
                  </div>
                )}

              {result && result.status === "OK" && result.class_scores && severity && (
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-[#8a94a3]">
                    Predicted category
                  </p>
                  <div className="mt-2 flex items-center gap-3">
                    <span
                      className={`rounded-md px-3 py-1 text-lg font-semibold ${severity.bg} ${severity.text}`}
                    >
                      {result.predicted_class}
                    </span>
                    <span className="text-sm text-[#6b7280]">{severity.label}</span>
                  </div>
                  <p className="mt-1 text-sm text-[#6b7280]">
                    {result.confidence !== null
                      ? `${(result.confidence * 100).toFixed(1)}% model confidence`
                      : ""}
                  </p>

                  <div className="mt-6 space-y-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-[#8a94a3]">
                      Class scores
                    </p>
                    {CLASS_ORDER.map((cls) => {
                      const score = result.class_scores?.[cls] ?? 0;
                      const isTop = cls === result.predicted_class;
                      return (
                        <div key={cls}>
                          <div className="mb-1 flex justify-between text-xs text-[#4b5563]">
                            <span className={isTop ? "font-semibold" : ""}>{cls}</span>
                            <span>{(score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full rounded-full bg-[#eef2f7]">
                            <div
                              className={`h-2 rounded-full ${isTop ? SEVERITY[cls].bar : "bg-[#c3cdd8]"}`}
                              style={{ width: `${score * 100}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Disclaimer */}
        <footer className="mt-8 flex items-start gap-2 rounded-lg border border-[#e2e8f0] bg-[#fafbfc] p-4 text-xs leading-relaxed text-[#6b7280]">
          <span className="mt-0.5 text-[#94a3b8]">ⓘ</span>
          <span>
            This tool provides computational decision support only and does not
            constitute a medical diagnosis. Results should be interpreted by an
            appropriately qualified healthcare professional.
          </span>
        </footer>
      </div>
    </main>
  );
}