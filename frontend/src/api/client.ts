export interface TrajectorySummary {
  trajectory_id: string;
  source_pipeline: string;
  label: string;
  step_count: number;
  started_at: number;
  completed_at: number | null;
}

export interface Step {
  step_id: string;
  step_type: string;
  agent_name: string;
  agent_class: string;
  input: unknown;
  output: unknown;
  started_at: number;
  completed_at: number;
  parent_step_id: string | null;
  declared_contract_ref: string | null;
  tap: string;
  evidence: Record<string, unknown>;
}

export interface TrajectoryDetail {
  trajectory_id: string;
  source_pipeline: string;
  source_spec_ref: string | null;
  steps: Step[];
  started_at: number;
  completed_at: number | null;
  outcome: unknown;
}

export interface CaseSummary {
  trajectory_id: string;
  goal: string;
  plan_declared: string[] | null;
  plan_actual: string[];
  outcome: unknown;
  step_type_map: Record<string, string>;
  orient_notes: string[];
}

export type Severity = "none" | "minor" | "significant" | "critical";

export interface Verdict {
  verdict_id: string;
  specialist: string;
  step_ids: string[];
  trigger_event: string;
  finding: string;
  severity: Severity;
  evidence: Record<string, unknown>;
}

export type RecommendationStatus = "proposed" | "approved" | "declined" | "held";

export interface Recommendation {
  recommendation_id: string;
  trajectory_id: string;
  tied_to_root_cause: boolean;
  root_cause_chain: string[];
  description: string;
  proposed_fix: string;
  status: RecommendationStatus;
}

export interface FactCheck {
  confidence: "high" | "medium" | "low";
  notes: string;
  routed_back: boolean;
}

export interface ReviewStatus {
  review_id: string;
  trajectory_id: string | null;
  status: "pending" | "running" | "complete" | "failed";
  error: string | null;
  stage_lineage_used: Record<string, string>;
  created_at: string;
  updated_at: string;
  case_summary?: CaseSummary | null;
  verdicts?: Verdict[];
  recommendations?: Recommendation[];
  fact_check?: FactCheck | null;
}

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp.json() as Promise<T>;
}

export function listTrajectories(): Promise<TrajectorySummary[]> {
  return request("/trajectories");
}

export function getTrajectory(trajectoryId: string): Promise<TrajectoryDetail> {
  return request(`/trajectories/${trajectoryId}`);
}

export function triggerReview(trajectoryId: string): Promise<{ review_id: string; status: string }> {
  return request(`/trajectories/${trajectoryId}/review`, { method: "POST" });
}

export function getReview(reviewId: string): Promise<ReviewStatus> {
  return request(`/reviews/${reviewId}`);
}

export function getConfig(): Promise<{ mode: "mock" | "live" }> {
  return request("/config");
}

export function applyGate(
  recommendationId: string,
  decision: "approved" | "declined" | "hold",
  reviewer: string,
  note?: string
): Promise<Recommendation> {
  return request(`/recommendations/${recommendationId}/gate`, {
    method: "POST",
    body: JSON.stringify({ decision, reviewer, note }),
  });
}
