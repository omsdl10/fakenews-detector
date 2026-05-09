/**
 * App — root component
 *
 * Layout:
 *   Header
 *   ArticleInput
 *   ── (after prediction) ──
 *   ResultCard  |  ExplanationHighlight
 *   EvidencePanel (full width)
 */

import { useState } from "react";
import ArticleInput from "./components/ArticleInput";
import ResultCard from "./components/ResultCard";
import EvidencePanel from "./components/EvidencePanel";
import ExplanationHighlight from "./components/ExplanationHighlight";
import { usePredict } from "./hooks/usePredict";
import "./App.css";

export default function App() {
  const { predict, result, loading, error, reset } = usePredict();
  const [submittedText, setSubmittedText] = useState("");

  async function handleSubmit({ text, url, deepExplain }) {
    setSubmittedText(text || "");
    await predict({ text, url, deepExplain });
  }

  return (
    <div className="app-container">
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🔍</span>
            <span className="logo-text">FakeNews Detector</span>
          </div>
          <p className="header-subtitle">
            AI-powered fake news classification with evidence retrieval and
            explainability
          </p>
        </div>
      </header>

      <main className="app-main">
        {/* ── Input ────────────────────────────────────────────────────────── */}
        <section className="input-section">
          <ArticleInput onSubmit={handleSubmit} loading={loading} />
        </section>

        {/* ── Error ────────────────────────────────────────────────────────── */}
        {error && (
          <div className="error-banner">
            <strong>Error:</strong> {error}
            <button onClick={reset} className="dismiss-btn">✕</button>
          </div>
        )}

        {/* ── Results ──────────────────────────────────────────────────────── */}
        {result && (
          <>
            <button onClick={reset} className="new-analysis-btn">
              ← New analysis
            </button>

            {/* Primary result + explanation side by side on wide screens */}
            <div className="results-grid">
              <ResultCard result={result} />
              {submittedText && (
                <ExplanationHighlight
                  articleText={submittedText}
                  explanation={result.explanation}
                  label={result.label}
                />
              )}
            </div>

            {/* Evidence full width below */}
            <EvidencePanel evidence={result.evidence} />
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Powered by RoBERTa · FAISS · FastAPI · React ·{" "}
          <a
            href="/api/v1/docs"
            target="_blank"
            rel="noopener noreferrer"
          >
            API Docs
          </a>
        </p>
      </footer>
    </div>
  );
}
