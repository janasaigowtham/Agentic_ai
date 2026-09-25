import { CaseSummary, FactCheck } from "../api/client";

interface Props {
  caseSummary: CaseSummary;
  factCheck: FactCheck | null;
  lineage: Record<string, string>;
}

export function CaseSummaryHeader({ caseSummary, factCheck, lineage }: Props) {
  return (
    <section className="case-summary">
      <h2>Case Summary</h2>
      <p className="goal">{caseSummary.goal}</p>
      <div className="lineage-row">
        {Object.entries(lineage).map(([stage, model]) => (
          <span key={stage} className={`lineage-badge lineage-${model}`}>
            {stage}: {model}
          </span>
        ))}
      </div>
      {caseSummary.orient_notes.length > 0 && (
        <div className="orient-notes">
          <h3>Orient notes (pointers, not verdicts)</h3>
          <ul>
            {caseSummary.orient_notes.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
        </div>
      )}
      {factCheck && (
        <div className={`fact-check confidence-${factCheck.confidence}`}>
          <strong>Fact-check ({factCheck.confidence} confidence):</strong> {factCheck.notes}
        </div>
      )}
    </section>
  );
}
