/**
 * ArticleInput
 * Accepts article text directly or a news URL.
 * Validates locally before calling onSubmit.
 */

import { useState } from "react";

const TAB = { TEXT: "text", URL: "url" };

export default function ArticleInput({ onSubmit, loading }) {
  const [activeTab, setActiveTab] = useState(TAB.TEXT);
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [deepExplain, setDeepExplain] = useState(false);
  const [validationError, setValidationError] = useState("");

  function validate() {
    if (activeTab === TAB.TEXT) {
      if (text.trim().length < 30) {
        setValidationError("Article text must be at least 30 characters.");
        return false;
      }
    } else {
      try {
        new URL(url.trim());
      } catch {
        setValidationError("Please enter a valid URL (include https://).");
        return false;
      }
    }
    setValidationError("");
    return true;
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!validate()) return;
    onSubmit({
      text: activeTab === TAB.TEXT ? text.trim() : undefined,
      url: activeTab === TAB.URL ? url.trim() : undefined,
      deepExplain,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="input-card">
      {/* Tab switcher */}
      <div className="tab-bar">
        {Object.values(TAB).map((tab) => (
          <button
            key={tab}
            type="button"
            className={`tab-btn ${activeTab === tab ? "active" : ""}`}
            onClick={() => {
              setActiveTab(tab);
              setValidationError("");
            }}
          >
            {tab === TAB.TEXT ? "Paste Article" : "Enter URL"}
          </button>
        ))}
      </div>

      {/* Input area */}
      {activeTab === TAB.TEXT ? (
        <textarea
          className="article-textarea"
          placeholder="Paste the full article text here…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
          maxLength={10000}
          disabled={loading}
        />
      ) : (
        <input
          type="url"
          className="url-input"
          placeholder="https://example.com/news/article"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading}
        />
      )}

      {/* Options row */}
      <div className="options-row">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={deepExplain}
            onChange={(e) => setDeepExplain(e.target.checked)}
            disabled={loading}
          />
          Deep explanation (SHAP — slower)
        </label>
        {activeTab === TAB.TEXT && (
          <span className="char-count">{text.length} / 10 000</span>
        )}
      </div>

      {validationError && (
        <p className="validation-error">{validationError}</p>
      )}

      <button type="submit" className="submit-btn" disabled={loading}>
        {loading ? (
          <>
            <span className="spinner" />
            Analysing…
          </>
        ) : (
          "Analyse Article"
        )}
      </button>
    </form>
  );
}
