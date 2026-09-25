import { Step, TrajectoryDetail, Verdict } from "../api/client";

interface Props {
  trajectory: TrajectoryDetail;
  verdicts: Verdict[];
}

const SEVERITY_ORDER = ["none", "minor", "significant", "critical"];

function worstSeverity(verdicts: Verdict[]): string {
  return verdicts.reduce(
    (acc, v) => (SEVERITY_ORDER.indexOf(v.severity) > SEVERITY_ORDER.indexOf(acc) ? v.severity : acc),
    "none"
  );
}

function hasDeclaredMismatch(step: Step): boolean {
  const ev = step.evidence ?? {};
  const undeclared = (ev.undeclared_template_refs as string[] | undefined) ?? [];
  const duplicates = (ev.duplicate_of_step_ids as string[] | undefined) ?? [];
  return undeclared.length > 0 || duplicates.length > 0;
}

export function TrajectoryStepViewer({ trajectory, verdicts }: Props) {
  const verdictsByStep = new Map<string, Verdict[]>();
  for (const v of verdicts) {
    for (const stepId of v.step_ids) {
      const list = verdictsByStep.get(stepId) ?? [];
      list.push(v);
      verdictsByStep.set(stepId, list);
    }
  }

  return (
    <section className="step-viewer">
      <h2>Trajectory Steps ({trajectory.steps.length})</h2>
      <ol className="step-list">
        {trajectory.steps.map((step) => {
          const mismatch = hasDeclaredMismatch(step);
          const stepVerdicts = verdictsByStep.get(step.step_id) ?? [];
          const worst = worstSeverity(stepVerdicts);
          const undeclared = (step.evidence?.undeclared_template_refs as string[] | undefined) ?? [];
          const duplicates = (step.evidence?.duplicate_of_step_ids as string[] | undefined) ?? [];
          return (
            <li
              key={step.step_id}
              className={`step-row severity-${worst}${mismatch ? " declared-mismatch" : ""}`}
            >
              <div className="step-row-main">
                <span className="step-id">{step.step_id}</span>
                <span className="step-name">{step.agent_name}</span>
                <span className="step-class">{step.agent_class}</span>
                <span className="step-tap" title="capture tap: a = pre-execution snapshot, b_only = after-the-fact only">
                  tap:{step.tap}
                </span>
                <span className="step-time">
                  {step.started_at.toFixed(3)}s &rarr; {step.completed_at.toFixed(3)}s
                </span>
              </div>
              {mismatch && (
                <div className="step-evidence-flags">
                  {undeclared.length > 0 && (
                    <span className="flag">undeclared refs: {undeclared.join(", ")}</span>
                  )}
                  {duplicates.length > 0 && (
                    <span className="flag">duplicate of: {duplicates.join(", ")}</span>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
