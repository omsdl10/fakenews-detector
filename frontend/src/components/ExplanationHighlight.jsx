/**
 * ExplanationHighlight
 * Renders the article text with tokens colour-highlighted by importance score.
 *
 * Heat map:
 *   score 0.0 → white  (no influence)
 *   score 1.0 → deep red/green depending on predicted label
 */

import { useMemo } from "react";

// Score → RGBA colour (label-aware)
function scoreToColor(score, label) {
  const alpha = Math.pow(score, 0.6); // gamma correction for readability
  if (label === "fake")
    return `rgba(229, 62, 62, ${alpha * 0.55})`;   // red family
  if (label === "real")
    return `rgba(56, 161, 105, ${alpha * 0.55})`;  // green family
  return `rgba(214, 158, 46, ${alpha * 0.55})`;    // amber for uncertain
}

/**
 * Merge token importance into text spans.
 * Tokens from the tokenizer may overlap (subword), so we render
 * character-level highlights using char_start/char_end when available,
 * otherwise do a simple word-level best-effort match.
 */
function buildHighlightedSpans(text, tokenImportance, label) {
  // Build a char-level score array
  const charScores = new Float32Array(text.length);

  for (const { token, score, char_start, char_end } of tokenImportance) {
    if (
      typeof char_start === "number" &&
      typeof char_end === "number" &&
      char_end > char_start
    ) {
      for (let i = char_start; i < Math.min(char_end, text.length); i++) {
        charScores[i] = Math.max(charScores[i], score);
      }
    } else {
      // Fallback: find first occurrence of the (cleaned) token word
      const cleaned = token.replace(/^Ġ/, "").trim();
      if (!cleaned) continue;
      const idx = text.toLowerCase().indexOf(cleaned.toLowerCase());
      if (idx !== -1) {
        for (let i = idx; i < Math.min(idx + cleaned.length, text.length); i++) {
          charScores[i] = Math.max(charScores[i], score);
        }
      }
    }
  }

  // Collapse consecutive chars with the same score into spans
  const THRESHOLD = 0.15; // below this → no highlight
  const spans = [];
  let i = 0;
  while (i < text.length) {
    const s = charScores[i];
    let j = i + 1;
    while (j < text.length && Math.abs(charScores[j] - s) < 0.05) j++;

    const segment = text.slice(i, j);
    if (s >= THRESHOLD) {
      spans.push({ text: segment, score: s, highlighted: true });
    } else {
      spans.push({ text: segment, score: 0, highlighted: false });
    }
    i = j;
  }
  return spans;
}

export default function ExplanationHighlight({ articleText, explanation, label }) {
  const spans = useMemo(
    () => buildHighlightedSpans(articleText, explanation.token_importance, label),
    [articleText, explanation, label]
  );

  return (
    <div className="explanation-card">
      <div className="section-title">
        Word Importance
        <span className="method-badge">{explanation.method}</span>
      </div>

      {/* Legend */}
      <div className="highlight-legend">
        <div className="legend-gradient" data-label={label} />
        <span>Low influence</span>
        <span style={{ marginLeft: "auto" }}>High influence</span>
      </div>

      {/* Highlighted text */}
      <div className="highlighted-text">
        {spans.map((span, idx) =>
          span.highlighted ? (
            <mark
              key={idx}
              title={`Score: ${(span.score * 100).toFixed(0)}%`}
              style={{
                background: scoreToColor(span.score, label),
                borderRadius: "2px",
                padding: "0 1px",
                cursor: "help",
              }}
            >
              {span.text}
            </mark>
          ) : (
            <span key={idx}>{span.text}</span>
          )
        )}
      </div>

      {/* Top tokens table */}
      <div className="section-title" style={{ marginTop: "1rem" }}>
        Top influential tokens
      </div>
      <div className="token-table">
        {explanation.token_importance.slice(0, 10).map((t, i) => (
          <div key={i} className="token-row">
            <span className="token-rank">{i + 1}</span>
            <code className="token-word">{t.token}</code>
            <div className="token-bar-track">
              <div
                className="token-bar-fill"
                style={{
                  width: `${t.score * 100}%`,
                  background: scoreToColor(t.score, label),
                }}
              />
            </div>
            <span className="token-score">{(t.score * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
