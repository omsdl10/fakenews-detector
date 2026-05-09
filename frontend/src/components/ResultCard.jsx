/**
 * ResultCard
 * Displays the primary classification result:
 *   - FAKE / REAL / UNCERTAIN label with color coding
 *   - Confidence percentage + animated progress bar
 *   - Credibility score (0–100) with gauge
 *   - Per-class probability breakdown
 */

const LABEL_META = {
  fake: { color: "#e53e3e", bg: "#fff5f5", emoji: "⚠️", text: "FAKE NEWS" },
  real: { color: "#38a169", bg: "#f0fff4", emoji: "✓", text: "LIKELY CREDIBLE" },
  uncertain: { color: "#d69e2e", bg: "#fffff0", emoji: "?", text: "UNCERTAIN" },
};

function CredibilityGauge({ score }) {
  const angle = (score / 100) * 180 - 90; // -90° (0) to +90° (100)
  const color =
    score < 35 ? "#e53e3e" : score < 65 ? "#d69e2e" : "#38a169";
  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 120 70" className="gauge-svg">
        {/* Background arc */}
        <path
          d="M 10 60 A 50 50 0 0 1 110 60"
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Score arc — we approximate by rotating the needle */}
        <path
          d="M 10 60 A 50 50 0 0 1 110 60"
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${(score / 100) * 157} 157`}
        />
        {/* Needle */}
        <line
          x1="60"
          y1="60"
          x2={60 + 38 * Math.cos(((angle - 90) * Math.PI) / 180)}
          y2={60 + 38 * Math.sin(((angle - 90) * Math.PI) / 180)}
          stroke="#4a5568"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="60" cy="60" r="4" fill="#4a5568" />
      </svg>
      <div className="gauge-score" style={{ color }}>
        {score.toFixed(0)}
        <span className="gauge-unit"> / 100</span>
      </div>
      <div className="gauge-label">Credibility Score</div>
    </div>
  );
}

function ProbBar({ label, prob }) {
  const meta = LABEL_META[label] || LABEL_META.uncertain;
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-track">
        <div
          className="prob-fill"
          style={{ width: `${prob * 100}%`, background: meta.color }}
        />
      </div>
      <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}

export default function ResultCard({ result }) {
  const meta = LABEL_META[result.label] || LABEL_META.uncertain;

  return (
    <div className="result-card" style={{ borderColor: meta.color }}>
      {/* Header */}
      <div className="result-header" style={{ background: meta.bg }}>
        <span className="result-emoji">{meta.emoji}</span>
        <div>
          <div className="result-label" style={{ color: meta.color }}>
            {meta.text}
          </div>
          <div className="result-confidence">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </div>
        </div>
        <CredibilityGauge score={result.credibility_score} />
      </div>

      {/* Probability breakdown */}
      <div className="prob-section">
        <div className="section-title">Probability breakdown</div>
        {Object.entries(result.probabilities).map(([label, prob]) => (
          <ProbBar key={label} label={label} prob={prob} />
        ))}
      </div>

      {/* Reasoning */}
      <div className="reasoning-section">
        <div className="section-title">Reasoning</div>
        <p className="reasoning-text">
          {result.explanation.reasoning_summary}
        </p>
      </div>

      {/* Metadata */}
      <div className="meta-row">
        <span>Prediction ID: {result.prediction_id.slice(0, 8)}…</span>
        <span>Latency: {result.latency_ms.toFixed(0)} ms</span>
      </div>
    </div>
  );
}
