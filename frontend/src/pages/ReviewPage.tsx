import { useEffect, useRef, useState } from "react";
import { getReview, getTrajectory, Recommendation, ReviewStatus, TrajectoryDetail } from "../api/client";
import { CaseSummaryHeader } from "../components/CaseSummaryHeader";
import { TrajectoryStepViewer } from "../components/TrajectoryStepViewer";
import { FindingsPanel } from "../components/FindingsPanel";
import { RecommendationGatePanel } from "../components/RecommendationGatePanel";

interface Props {
  trajectoryId: string;
  reviewId: string;
  onBack: () => void;
}

export function ReviewPage({ trajectoryId, reviewId, onBack }: Props) {
  const [trajectory, setTrajectory] = useState<TrajectoryDetail | null>(null);
  const [review, setReview] = useState<ReviewStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    getTrajectory(trajectoryId)
      .then(setTrajectory)
      .catch((e) => setError(String(e)));
  }, [trajectoryId]);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const status = await getReview(reviewId);
        if (cancelled) return;
        setReview(status);
        if (status.status === "complete" || status.status === "failed") {
          return;
        }
      } catch (e) {
        if (!cancelled) setError(String(e));
        return;
      }
      pollRef.current = window.setTimeout(poll, 1000);
    }

    poll();
    return () => {
      cancelled = true;
      if (pollRef.current) window.clearTimeout(pollRef.current);
    };
  }, [reviewId]);

  function handleRecommendationChanged(updated: Recommendation) {
    setReview((prev) =>
      prev
        ? {
            ...prev,
            recommendations: (prev.recommendations ?? []).map((r) =>
              r.recommendation_id === updated.recommendation_id ? updated : r
            ),
          }
        : prev
    );
  }

  return (
    <div className="page">
      <header className="page-header">
        <button className="link-button" onClick={onBack}>
          &larr; Back to records
        </button>
        <h1>Review {reviewId}</h1>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {!review || review.status === "pending" || review.status === "running" ? (
        <div className="status-banner">Running judgment pipeline ({review?.status ?? "pending"})&hellip;</div>
      ) : review.status === "failed" ? (
        <div className="error-banner">Review failed: {review.error}</div>
      ) : (
        <>
          {review.case_summary && (
            <CaseSummaryHeader
              caseSummary={review.case_summary}
              factCheck={review.fact_check ?? null}
              lineage={review.stage_lineage_used}
            />
          )}
          {trajectory && <TrajectoryStepViewer trajectory={trajectory} verdicts={review.verdicts ?? []} />}
          <FindingsPanel verdicts={review.verdicts ?? []} recommendations={review.recommendations ?? []} />
          <RecommendationGatePanel
            recommendations={review.recommendations ?? []}
            onChanged={handleRecommendationChanged}
          />
        </>
      )}
    </div>
  );
}
