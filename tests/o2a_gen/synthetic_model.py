"""A scripted model for smoke tests: answers every o2a_gen prompt from the prompt itself.

It knows nothing about any particular procedure, tools or syntax. It reads the
agent classes, procedure headings, session keys, branch targets and required
keys out of each prompt, so it works on any inputs. It also injects the failures
the real API produces, so the recovery paths run: the first summary card is
declined by the main model (answered by the fallback), the first extraction
stream breaks mid-way, and the first answer for an LLM-judgement agent references
a session key that does not exist (the check must send it back).
"""

from __future__ import annotations

import json
import re
import threading

from o2a_gen.ground import GROUND_SYSTEM
from o2a_gen.navigator import NAV_SYSTEM
from o2a_gen.procedure import EXTRACT_SYSTEM
from o2a_gen.proctree import SECTION_SYSTEM
from o2a_gen.skilltree import CARD_SYSTEM, FOLDER_SYSTEM


def _text(block) -> str:
    return block if isinstance(block, str) else "".join(b.get("text", "") for b in block)


class SyntheticModel:
    def __init__(self, main_model: str = "claude-opus-5-5"):
        self.main = main_model
        self.lock = threading.Lock()
        self.flags = {"refused": False, "broke": False}
        self.bad_llm_answers: set[str] = set()
        self.counts: dict[str, int] = {}

    def __call__(self, body: dict):
        system = _text(body.get("system") or "")
        msgs = [_text(m["content"]) for m in body["messages"]]
        kind = {CARD_SYSTEM: "card", SECTION_SYSTEM: "section", FOLDER_SYSTEM: "folder",
                EXTRACT_SYSTEM: "extract", NAV_SYSTEM: "navigate",
                GROUND_SYSTEM: "ground"}.get(system, "other")
        with self.lock:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            if kind == "card" and not self.flags["refused"] and body["model"] == self.main:
                self.flags["refused"] = True
                return {"text": "", "stop_reason": "refusal"}
            if kind == "extract" and not self.flags["broke"]:
                self.flags["broke"] = True
                return {"stream_error": "overloaded_error"}
        answer = getattr(self, "_" + kind)(msgs)
        if kind == "extract":   # a long answer, streamed slowly, like the real one
            return {"text": json.dumps(answer), "slow": 0.02, "pings": 25}
        return json.dumps(answer) if not isinstance(answer, str) else answer

    # ------------------------------------------------------------ compile time
    def _card(self, msgs):
        name = re.search(r"\[\w+\] (.+)", msgs[-1]).group(1).strip()
        return {"title": name, "one_line": f"About {name}.",
                "keywords": list(dict.fromkeys(re.findall(r"\b[A-Za-z_][\w]{3,}\b", msgs[-1])))[:8]}

    def _section(self, msgs):
        title = msgs[-1].split("\n", 1)[0].replace("Section:", "").strip()
        return {"summary": f"This part covers {title}."}

    def _folder(self, msgs):
        first = re.search(r"- (?:sub-folder ')?([^:'\n]+)", msgs[-1])
        label = re.sub(r"[^a-z0-9]+", "-", (first.group(1) if first else "items").lower()).strip("-")
        return {"label": label[:40] or "items", "summary": f"Items like {label}."}

    def _other(self, msgs):
        return {"ok": True}

    # ------------------------------------------------------------ extraction
    def _extract(self, msgs):
        prompt = msgs[0]
        classes = re.search(r'"agent_class": one of (.+?),\n', prompt).group(1).split(" | ")
        pick = lambda *words: next((c for w in words for c in classes if w in c.lower()), None)
        router, gate = pick("router"), pick("gate")
        orchestrator = pick("orchestrator") or pick("sequential")
        group = pick("sequential") or orchestrator
        workers = [c for c in (pick("database"), pick("rest"), pick("slv"), pick("llm"),
                               pick("transformation")) if c]
        flagger = pick("slv", "transformation") or workers[0]
        numbered = re.findall(r"^(\d+)\| (.*)$", prompt.split("Procedure (each line")[1], re.M)
        n_lines = len(numbered)
        heads = [(int(n), len(t) - len(t.lstrip("#")), t.lstrip("#").strip())
                 for n, t in numbered if re.match(r"#{1,6} ", t)] or [(1, 1, "Procedure")]
        top = min(h[1] for h in heads)
        phase_level = top + 1 if any(h[1] == top + 1 for h in heads) else top

        def span(i):
            nxt = next((heads[j][0] for j in range(i + 1, len(heads))), n_lines + 1)
            return f"{heads[i][0]}-{max(heads[i][0], nxt - 1)}"

        phases, sid = [], 0
        for i, (line, level, title) in enumerate(heads):
            if level == phase_level:
                phases.append({"id": f"P{len(phases) + 1}", "title": title[:60], "steps": [],
                               "_line": f"{line}-{line}"})
            elif level > phase_level and phases:
                phases[-1]["steps"].append((title, span(i)))
        phases = [p for p in phases if p["steps"]] or [
            {"id": "P1", "title": heads[0][2], "steps": [(heads[0][2], span(0))], "_line": "1-1"}]

        out_phases, prev = [], None
        for p in phases:
            steps = []
            for title, lines in p["steps"][:6]:
                sid += 1
                cls = workers[(sid - 1) % len(workers)]
                steps.append({"id": f"S{sid}", "title": title[:60], "text": f"Do: {title}",
                              "source_lines": lines, "agent_class": cls,
                              "produces": f"{title[:30]} result",
                              "uses": [prev] if prev else ["loan_number"]})
                prev = f"S{sid}"
            out_phases.append({"id": p["id"], "title": p["title"], "direct": False, "steps": steps})

        # one branching step (flag, then router with two single-step branches)
        if router and out_phases:
            ph = max(out_phases, key=lambda q: len(q["steps"]))
            lines = ph["steps"][-1]["source_lines"]
            prev = ph["steps"][-1]["id"]           # the flag reads the phase's last step
            flag, br = f"S{sid + 1}", f"S{sid + 2}"
            ph["steps"] += [
                {"id": flag, "title": "Exception flag", "text": "Y when an exception applies.",
                 "source_lines": lines, "agent_class": flagger, "produces": "exception flag",
                 "uses": [prev]},
                {"id": br, "title": "Exception decision", "text": "Branch on the flag.",
                 "source_lines": lines, "agent_class": router, "depends_on": flag, "uses": [flag],
                 "branches": [
                     {"label": "exception", "when": "flag is Y", "steps": [
                         {"id": f"S{sid + 3}", "title": "Review exception", "text": "Review it.",
                          "source_lines": lines, "agent_class": pick("llm") or flagger,
                          "produces": "review outcome", "uses": [prev]}]},
                     {"label": "no exception", "when": "flag is N", "steps": [
                         {"id": f"S{sid + 4}", "title": "Record no exception", "text": "Record.",
                          "source_lines": lines, "agent_class": flagger,
                          "produces": "review outcome", "uses": []}]}]}]
            sid += 4
        if gate:
            sid += 1
            out_phases.append({"id": f"P{len(out_phases) + 1}", "title": "Approval", "direct": True,
                               "steps": [{"id": f"S{sid}", "title": "Approval gate",
                                          "text": "Wait for approval.",
                                          "source_lines": out_phases[-1]["steps"][0]["source_lines"],
                                          "agent_class": gate, "produces": "approval status",
                                          "uses": []}]})
        return {"name": "synthetic_review", "description": "Synthetic smoke-test workflow.",
                "inputs": ["loan_number"], "orchestrator_class": orchestrator,
                "group_class": group, "phases": out_phases}

    # ------------------------------------------------------------ serve time
    def _navigate(self, msgs):
        turn = (len(msgs) + 1) // 2
        if turn == 1:
            words = re.findall(r"[A-Za-z]{5,}", msgs[0].split("What I need:")[1])
            return {"action": "find", "term": (words[2] if len(words) > 2 else "data").lower()}
        ids = re.findall(r"`([a-z0-9]{8,})`", msgs[-1])
        if turn == 2 and ids:
            return {"action": "get_document", "doc_id": ids[0]}
        found = re.findall(r"`([a-z0-9]{8,})`", "\n".join(msgs[1:]))
        return {"action": "done", "doc_ids": list(dict.fromkeys(found))[:3],
                "notes": "synthetic navigation"}

    def _ground(self, msgs):
        prompt = msgs[0]
        name, cls = re.search(r"Agent to write: `([^`]+)` \(([^)]+)\)", prompt).groups()
        avail = json.loads(re.search(r"Session keys available when it runs: (\[.*?\])",
                                     prompt).group(1).replace("'", '"'))
        syntax = prompt.split("Agent to write:")[0]
        targets = re.findall(r"^- .+?: when .+? -> (\S+)$", prompt, re.M)
        fields: dict = {}
        req = re.search(r"### Required keys\n((?:- .+\n)+)", syntax)
        for key in re.findall(r"- (\w+)", req.group(1)) if req else []:
            if key not in ("name", "agent_class", "output_key", "sub_agents", "routes"):
                fields[key] = ""                              # value not in the inputs
        if "input_keys" in syntax and avail:
            fields["input_keys"] = avail[-1:]
        if targets:
            fields["routes"] = [{"conditions": [{"context_key": avail[-1], "operator": "eq",
                                                 "value": v}], "target_agent": t}
                                for t, v in zip(targets, ("Y", "N", "X", "Z"))]
        if "llm" in cls.lower():
            with self.lock:
                first = name not in self.bad_llm_answers
                self.bad_llm_answers.add(name)
            ref = "{no_such_key}" if first else "{" + (avail[-1] if avail else "x") + "}"
            fields["instruction"] = f"Review {ref} and report the outcome."
        gaps = [f"{k}: not in the inputs" for k, v in fields.items() if v == ""]
        return {"fields": fields, "description": f"{cls} for {name}.", "gaps": gaps}
