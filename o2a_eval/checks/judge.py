"""On-prem LLM-as-judge client, and the two semantic checks that use it: N4, N5."""
from __future__ import annotations

import json
import re
import urllib.request
from urllib.parse import urlparse

from ..core.models import Flaw, Severity
from ..core.registry import register

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_FIRST_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeEndpointError(RuntimeError):
    """Raised when a judge endpoint is not local. Never caught by callers."""


class LocalJudgeClient:
    ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

    def __init__(self, endpoint: str, model: str, allowed_hosts=None, timeout: int = 120, max_calls: int = 10):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.max_calls = max_calls
        self.calls_used = 0
        allowed = set(allowed_hosts) if allowed_hosts else set(self.ALLOWED_HOSTS)

        hostname = (urlparse(endpoint).hostname or "").lower()
        is_local = hostname in allowed or hostname.endswith(".internal") or hostname.endswith(".local")
        if not is_local:
            raise JudgeEndpointError(
                f"refusing non-local judge endpoint {endpoint!r} (host {hostname!r})"
            )

    @property
    def exhausted(self) -> bool:
        return self.calls_used >= self.max_calls

    def complete(self, prompt: str) -> str:
        if self.exhausted:
            raise RuntimeError(f"judge call budget exhausted ({self.max_calls})")
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            self.endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        self.calls_used += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]

    def complete_json(self, prompt: str) -> dict:
        text = self.complete(prompt)
        fence = _JSON_FENCE_RE.search(text)
        candidate = fence.group(1) if fence else text
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            pass
        match = _FIRST_OBJECT_RE.search(text)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"could not parse JSON from judge response: {text[:200]!r}")


N5_RUBRIC = """Node: {node}
Declared role: {role}
Declared inputs: {inputs}
Declared output key: {output_key}
Downstream consumers: {downstream}

Input received:
{input_value}

Output produced:
{output_value}

Score 1-5 on each axis with one sentence reason:
1. role_fulfillment: did output fulfil the declared role?
2. input_utilization: did it use declared inputs or re-derive them?
3. output_completeness: is output well-formed for downstream consumers?
4. scope_adherence: did it stay within declared role?

Also return failure_mode, one of:
none | wrong_role | incomplete_output | ignored_input | scope_leak | hallucinated_output | other

Return strict JSON only:
{{"role_fulfillment":{{"score":int,"reason":str}},"input_utilization":{{"score":int,"reason":str}},
"output_completeness":{{"score":int,"reason":str}},"scope_adherence":{{"score":int,"reason":str}},
"failure_mode":str}}"""


@register("N5", axis="output_quality", stage="runtime", needs_judge=True)
def judge_output_quality(run, graph, config):
    """LlmAgent output judged against its declared role and inputs."""
    judge = config.get("_judge")
    if judge is None:
        return []

    flaws: list[Flaw] = []
    for span in run.llm_spans():
        if judge.exhausted:
            break
        node = graph.nodes.get(span.agent_name)
        if node is None:
            continue

        prompt = N5_RUBRIC.format(
            node=node.name,
            role=node.description or "(no description)",
            inputs=node.input_keys,
            output_key=node.output_key or "",
            downstream=graph.consumers.get(node.output_key or "", []),
            input_value=str(span.input_value)[:4000],
            output_value=str(span.output_value)[:4000],
        )
        try:
            verdict = judge.complete_json(prompt)
        except Exception as e:  # noqa: BLE001
            flaws.append(
                Flaw(
                    type="judge_error",
                    severity=Severity.INFO,
                    node=node.name,
                    description=f"N5 judge call failed for {node.name}: {type(e).__name__}: {e}",
                    fix=None,
                    evidence={},
                    agent_class=node.agent_class,
                )
            )
            continue

        axes = {
            k: verdict.get(k, {})
            for k in ("role_fulfillment", "input_utilization", "output_completeness", "scope_adherence")
        }
        scores = [v.get("score") for v in axes.values() if isinstance(v.get("score"), (int, float))]
        failure_mode = verdict.get("failure_mode")
        if not scores and failure_mode in (None, "none"):
            continue
        worst = min(scores) if scores else 5
        flagged = worst <= 3 or failure_mode not in (None, "none")
        if not flagged:
            continue

        if worst <= 2:
            severity = Severity.SEV_1
        elif worst == 3:
            severity = Severity.SEV_2
        else:
            severity = Severity.SEV_2

        mean_score = sum(scores) / len(scores) if scores else None
        flaws.append(
            Flaw(
                type="judged_output_quality_issue",
                severity=severity,
                node=node.name,
                description=f"{node.name}'s output scored poorly on judged quality (failure_mode={failure_mode}).",
                fix="Review the prompt for role clarity, input usage, and output format.",
                evidence={"axes": axes, "failure_mode": failure_mode, "mean_score": mean_score},
                agent_class=node.agent_class,
            )
        )
    return flaws


def _describe_field(key: str, val) -> str:
    if isinstance(val, dict):
        return f"{key}: object with keys {sorted(val)[:15]}"
    if isinstance(val, list) and val and isinstance(val[0], dict):
        return f"{key}: list[{len(val)}] of objects with keys {sorted(val[0])[:15]}"
    return f"{key}: {type(val).__name__}"


N4_RUBRIC = """Session already contains these structured fields:
{field_descriptions}

Prompt sent to model:
{rendered_prompt}

Does the prompt instruct the model to extract, parse, derive or infer any value
that already exists as a structured field above?

Return strict JSON only:
{{"bypasses": bool, "fields": [str], "reason": str}}"""


@register("N4", axis="prompt_quality", stage="runtime", needs_judge=True)
def judge_redundant_derivation(run, graph, config):
    """Prompt asks the model to re-derive a value already present as structured data."""
    judge = config.get("_judge")
    if judge is None:
        return []

    flaws: list[Flaw] = []
    for span in run.llm_spans():
        if judge.exhausted:
            break
        node = graph.nodes.get(span.agent_name)
        if node is None:
            continue
        snapshot = span.session_snapshot
        if not snapshot or not span.rendered_prompt:
            continue

        field_descriptions = "\n".join(_describe_field(k, v) for k, v in snapshot.items())
        prompt = N4_RUBRIC.format(
            field_descriptions=field_descriptions,
            rendered_prompt=span.rendered_prompt[:6000],
        )
        try:
            verdict = judge.complete_json(prompt)
        except Exception as e:  # noqa: BLE001
            flaws.append(
                Flaw(
                    type="judge_error",
                    severity=Severity.INFO,
                    node=node.name,
                    description=f"N4 judge call failed for {node.name}: {type(e).__name__}: {e}",
                    fix=None,
                    evidence={},
                    agent_class=node.agent_class,
                )
            )
            continue

        fields = verdict.get("fields") or []
        if not verdict.get("bypasses") or not fields:
            continue

        flaws.append(
            Flaw(
                type="judged_redundant_derivation",
                severity=Severity.SEV_2,
                node=node.name,
                description=f"{node.name}'s prompt asks the model to re-derive {fields} from prose.",
                fix="Pass the structured field directly instead of asking the model to re-derive it.",
                evidence={
                    "re_derived_fields": fields,
                    "available_field_names": sorted(snapshot.keys()),
                    "judge_reason": str(verdict.get("reason", ""))[:300],
                },
                agent_class=node.agent_class,
            )
        )
    return flaws
