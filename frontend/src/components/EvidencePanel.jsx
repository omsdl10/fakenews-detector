/**
 * EvidencePanel
 * Shows semantically similar articles retrieved from the corpus,
 * tagged as "supports", "contradicts", or "inconclusive".
 */

const RELATION_META = {
  supports: { label: "Supports", color: "#38a169", bg: "#f0fff4", icon: "↑" },
  contradicts: { label: "Contradicts", color: "#e53e3e", bg: "#fff5f5", icon: "↓" },
  inconclusive: { label: "Inconclusive", color: "#718096", bg: "#f7fafc", icon: "–" },
};

function SimilarityBar({ score }) {
  return (
    <div className="similarity-bar-wrap" title={`Similarity: ${(score * 100).toFixed(0)}%`}>
      <div
        className="similarity-bar"
        style={{ width: `${score * 100}%` }}
      />
      <span className="similarity-pct">{(score * 100).toFixed(0)}%</span>
    </div>
  );
}

function EvidenceCard({ item }) {
  const rel = RELATION_META[item.relation] || RELATION_META.inconclusive;

  return (
    <div className="evidence-item" style={{ borderLeftColor: rel.color }}>
      <div className="evidence-header">
        <span className="relation-badge" style={{ color: rel.color, background: rel.bg }}>
          {rel.icon} {rel.label}
        </span>
        <SimilarityBar score={item.score} />
      </div>

      {item.title && <div className="evidence-title">{item.title}</div>}

      <p className="evidence-snippet">{item.text_snippet}</p>

      <div className="evidence-meta">
        {item.source_url ? (
          <a
            href={item.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="evidence-link"
          >
            {item.source_url.length > 60
              ? item.source_url.slice(0, 60) + "…"
              : item.source_url}
          </a>
        ) : (
          <span className="evidence-no-url">No source URL</span>
        )}
      </div>
    </div>
  );
}

export default function EvidencePanel({ evidence }) {
  if (!evidence || evidence.length === 0) {
    return (
      <div className="evidence-card empty">
        <div className="section-title">Supporting Evidence</div>
        <p className="empty-msg">
          No similar articles found in the corpus. Upload more articles to
          build a richer evidence base.
        </p>
      </div>
    );
  }

  const contradicts = evidence.filter((e) => e.relation === "contradicts");
  const supports = evidence.filter((e) => e.relation === "supports");
  const inconclusive = evidence.filter((e) => e.relation === "inconclusive");

  return (
    <div className="evidence-card">
      <div className="section-title">
        Supporting Evidence
        <span className="evidence-count">{evidence.length} articles</span>
      </div>

      {contradicts.length > 0 && (
        <div className="evidence-group">
          <div className="evidence-group-label" style={{ color: "#e53e3e" }}>
            Contradicting sources ({contradicts.length})
          </div>
          {contradicts.map((e) => (
            <EvidenceCard key={e.faiss_id} item={e} />
          ))}
        </div>
      )}

      {supports.length > 0 && (
        <div className="evidence-group">
          <div className="evidence-group-label" style={{ color: "#38a169" }}>
            Supporting sources ({supports.length})
          </div>
          {supports.map((e) => (
            <EvidenceCard key={e.faiss_id} item={e} />
          ))}
        </div>
      )}

      {inconclusive.length > 0 && (
        <div className="evidence-group">
          <div className="evidence-group-label" style={{ color: "#718096" }}>
            Related / inconclusive ({inconclusive.length})
          </div>
          {inconclusive.map((e) => (
            <EvidenceCard key={e.faiss_id} item={e} />
          ))}
        </div>
      )}
    </div>
  );
}
