import { useState } from "react";
import { RecordListPage } from "./pages/RecordListPage";
import { ReviewPage } from "./pages/ReviewPage";

export function App() {
  const [trajectoryId, setTrajectoryId] = useState<string | null>(null);
  const [reviewId, setReviewId] = useState<string | null>(null);

  if (trajectoryId && reviewId) {
    return (
      <ReviewPage
        trajectoryId={trajectoryId}
        reviewId={reviewId}
        onBack={() => {
          setTrajectoryId(null);
          setReviewId(null);
        }}
      />
    );
  }

  return (
    <RecordListPage
      onReviewStarted={(tId, rId) => {
        setTrajectoryId(tId);
        setReviewId(rId);
      }}
    />
  );
}
