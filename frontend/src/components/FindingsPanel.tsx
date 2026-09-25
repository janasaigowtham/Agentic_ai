import { Recommendation, Verdict } from "../api/client";

interface Props {
  verdicts: Verdict[];
  recommendations: Recommendation[];
}

const SEVERITY_ORDER = ["critical", "significant", "minor", "none"];

export function FindingsPanel({ verdicts, recommendations }: Props) {
  const rootCauseStepIds = new Set(recommendations.flatMap((r) => r.root_cause_chain));

  // Root-cause-linked findings surface first for triage; everything else
  // stays fully present underneath -- an ordering rule, never a filter.
  const sorted = [...verdicts].sort((a, b) => {
    const aRoot = a.step_ids.some((id) => rootCauseStepIds.has(id));
    const bRoot = b.step_ids.some((id) => rootCauseStepIds.has(id));
    if (aRoot !== bRoot) return aRoot ? -1 : 1;
    return SEVERITY_ORDER.indexOf(a.severity) - SEVERITY_ORDER.indexOf(b.severity);
  });

  return (
    <section className="findings-panel">
      <h2>Findings ({verdicts.length}, none dropped)</h2>
      <ul className="finding-list">
        {sorted.map((v) => {
          const isRootCause = v.step_ids.some((id) => rootCauseStepIds.has(id));
          return (
            <li key={v.verdict_id} className={`finding severity-${v.severity}${isRootCause ? " root-cause" : ""}`}>
              <div className="finding-header">
                <span className="specialist-badge">{v.specialist}</span>
                <span className={`severity-badge severity-${v.severity}`}>{v.severity}</span>
                {isRootCause && <span className="root-cause-badge">root cause</span>}
                <span className="finding-steps">{v.step_ids.join(", ")}</span>
              </div>
              <p className="finding-text">{v.finding}</p>
              <p className="trigger-event">Triggered by: {v.trigger_event}</p>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
