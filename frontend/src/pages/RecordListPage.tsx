import { useEffect, useState } from "react";
import { getConfig, listTrajectories, triggerReview, TrajectorySummary } from "../api/client";

interface Props {
  onReviewStarted: (trajectoryId: string, reviewId: string) => void;
}

export function RecordListPage({ onReviewStarted }: Props) {
  const [trajectories, setTrajectories] = useState<TrajectorySummary[]>([]);
  const [mode, setMode] = useState<string>("...");
  const [startingId, setStartingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listTrajectories()
      .then(setTrajectories)
      .catch((e) => setError(String(e)));
    getConfig()
      .then((c) => setMode(c.mode))
      .catch(() => setMode("unknown"));
  }, []);

  async function handleReview(trajectoryId: string) {
    setStartingId(trajectoryId);
    setError(null);
    try {
      const { review_id } = await triggerReview(trajectoryId);
      onReviewStarted(trajectoryId, review_id);
    } catch (e) {
      setError(String(e));
    } finally {
      setStartingId(null);
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <h1>Trajectory Review Harness</h1>
        <span className={`mode-badge mode-${mode}`}>{mode === "live" ? "LIVE" : "MOCK"}</span>
      </header>
      {error && <div className="error-banner">{error}</div>}
      <table className="record-table">
        <thead>
          <tr>
            <th>Trajectory</th>
            <th>Pipeline</th>
            <th>Steps</th>
            <th>Duration</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {trajectories.map((t) => (
            <tr key={t.trajectory_id}>
              <td>{t.label}</td>
              <td>{t.source_pipeline}</td>
              <td>{t.step_count}</td>
              <td>{t.completed_at !== null ? `${(t.completed_at - t.started_at).toFixed(1)}s` : "-"}</td>
              <td>
                <button disabled={startingId === t.trajectory_id} onClick={() => handleReview(t.trajectory_id)}>
                  {startingId === t.trajectory_id ? "Starting..." : "Run review"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
