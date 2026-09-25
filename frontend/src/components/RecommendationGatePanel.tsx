import { useState } from "react";
import { applyGate, Recommendation } from "../api/client";

interface Props {
  recommendations: Recommendation[];
  onChanged: (updated: Recommendation) => void;
}

export function RecommendationGatePanel({ recommendations, onChanged }: Props) {
  const [reviewer, setReviewer] = useState("demo-reviewer");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function decide(recommendationId: string, decision: "approved" | "declined" | "hold") {
    setBusyId(recommendationId);
    setError(null);
    try {
      const updated = await applyGate(recommendationId, decision, reviewer);
      onChanged(updated);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <section className="gate-panel">
      <h2>Recommendations ({recommendations.length})</h2>
      <label className="reviewer-input">
        Reviewer:
        <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
      </label>
      {error && <div className="error-banner">{error}</div>}
      <ul className="recommendation-list">
        {recommendations.map((r) => (
          <li key={r.recommendation_id} data-rec-id={r.recommendation_id} className={`recommendation status-${r.status}`}>
            <div className="recommendation-header">
              {r.tied_to_root_cause && <span className="root-cause-badge">root cause</span>}
              <span className="status-badge">{r.status}</span>
              <span className="chain">steps: {r.root_cause_chain.join(", ") || "-"}</span>
            </div>
            <p className="description">{r.description}</p>
            <p className="proposed-fix">
              <strong>Proposed fix:</strong> {r.proposed_fix}
            </p>
            <div className="gate-actions">
              <button
                disabled={busyId === r.recommendation_id || r.status !== "proposed"}
                onClick={() => decide(r.recommendation_id, "approved")}
              >
                Approve
              </button>
              <button
                disabled={busyId === r.recommendation_id || r.status !== "proposed"}
                onClick={() => decide(r.recommendation_id, "declined")}
              >
                Decline
              </button>
              <button
                disabled={busyId === r.recommendation_id || r.status !== "proposed"}
                onClick={() => decide(r.recommendation_id, "hold")}
              >
                Hold
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
