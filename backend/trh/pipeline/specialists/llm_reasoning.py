from trh.core.harness_models import CaseSummary, Step, Trajectory, Verdict
from trh.judge.client import JudgeClient
from trh.pipeline.specialists.base import Specialist
from trh.pipeline.text_fingerprint import sentence_fingerprints


class LlmReasoningSpecialist(Specialist):
    name = "llm_reasoning"
    agent_classes = frozenset({"LlmAgent"})
    focus = (
        "whether the prompt only draws on what its declared input_keys allow, "
        "whether its instruction leaks data the step shouldn't have needed "
        "(evidence.undeclared_template_refs), and whether the output is "
        "actually grounded in the structured fields it was given"
    )

    def evaluate(
        self,
        step: Step,
        trajectory: Trajectory,
        case_summary: CaseSummary,
        judge: JudgeClient,
    ) -> list[Verdict]:
        verdicts = super().evaluate(step, trajectory, case_summary, judge)
        span = step.raw_span
        if span is not None and span.rendered_prompt:
            # Plain code, pure plumbing: attaches which long prompt sentences
            # this step shares with others, for the Aggregator to reason
            # about across steps. This specialist does not compare itself.
            fingerprints = sentence_fingerprints(span.rendered_prompt)
            for verdict in verdicts:
                verdict.evidence = dict(verdict.evidence)
                verdict.evidence["prompt_sentence_fingerprints"] = fingerprints
        return verdicts
