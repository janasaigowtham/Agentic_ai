import json

import pytest

from o2a_eval import checks  # noqa: F401 - registers checks
from o2a_eval.checks.structural import error_swallowed, sequential_order, transform_ref_unresolved
from o2a_eval.core.graph import extract_sql_params, extract_template_keys, load_pipeline
from o2a_eval.core.models import Flaw, Node, Run, Severity, Span
from o2a_eval.core.rollup import build_scorecard
from o2a_eval.report.scrubber import scrub_evidence, scorecard_to_safe_dict, trace_summary
from o2a_eval.capture.ephemeral import EphemeralTraceStore


class TestGraph:
    def test_all_nodes_resolved_no_missing(self, graph):
        missing = [n for n in graph.nodes.values() if n.agent_class == "<missing>"]
        assert missing == []

    def test_non_deterministic_classification(self, graph):
        llm_names = {n.name for n in graph.non_deterministic()}
        assert llm_names == {"pmi_ddn_qa_q25527", "pmi_ddn_verdict_synthesizer"}

    def test_producer_map(self, graph):
        assert graph.producers["pmi_ddn_msp_loan_data"] == "pmi_ddn_fetch_msp_loan_data"
        assert graph.producers["pmi_ddn_final_output"] == "pmi_ddn_verdict_synthesizer"

    def test_consumer_map_includes_instruction_refs(self, graph):
        assert "pmi_ddn_qa_q25527" in graph.consumers["pmi_ddn_msp_loan_data"]

    def test_db_yaml_parsed_from_embedded_string(self, graph):
        node = graph.nodes["pmi_ddn_fetch_msp_loan_data"]
        assert node.db_type == "teradata"
        assert "MSP_LOAN_MASTER7_CS" in node.query
        assert node.db_url == "${ENV:ECRM_DB_CONN}"

    def test_db_yaml_parsed_from_nested_dict(self, graph):
        node = graph.nodes["pmi_ddn_fetch_investor_data"]
        assert node.db_type == "postgres"
        assert "investor_master" in node.query

    def test_route_parsing(self, graph):
        router = graph.nodes["pmi_ddn_icmp_decision_router"]
        assert len(router.routes) == 2
        targets = {r.target_agent for r in router.routes}
        assert targets == {"pmi_ddn_icmp_process", "pmi_ddn_no_icmp_path_handler"}
        assert router.routes[0].context_keys == ["pmi_ddn_icmp_required_flag"]

    def test_pipeline_hash_stable(self):
        g1 = load_pipeline("tests/fixtures/agents", "pmi_ddn_pipeline")
        g2 = load_pipeline("tests/fixtures/agents", "pmi_ddn_pipeline")
        assert g1.pipeline_hash == g2.pipeline_hash

    def test_unknown_pipeline_raises(self):
        with pytest.raises(ValueError):
            load_pipeline("tests/fixtures/agents", "does_not_exist")


class TestExtractors:
    def test_double_brace(self):
        assert extract_template_keys("{{ foo }}") == {"foo"}

    def test_double_brace_with_index_and_field(self):
        assert extract_template_keys("{{ foo[0].field }}") == {"foo"}

    def test_single_brace(self):
        assert extract_template_keys("value is {foo}") == {"foo"}

    def test_nested_structure(self):
        blob = {"a": ["{{ x }}", {"b": "{y}"}], "c": None}
        assert extract_template_keys(blob) == {"x", "y"}

    def test_sql_param_extraction(self):
        assert extract_sql_params("WHERE a = :foo AND b = :bar") == {"foo", "bar"}


class TestTrace:
    def test_tree_reconstruction(self, run):
        assert run.root is not None
        assert run.root.agent_name == "pmi_ddn_pipeline"
        assert len(run.root.children) == 4

    def test_span_typed_properties(self, run):
        router = run.spans_for("pmi_ddn_icmp_decision_router")[0]
        assert router.agent_class == "decision_router_agent"
        assert router.chosen_target == "pmi_ddn_icmp_process"
        assert sum(1 for r in router.evaluated_routes if r["matched"]) == 1

        vs = run.spans_for("pmi_ddn_verdict_synthesizer")[0]
        assert "pmi_ddn_msp_loan_data" in vs.template_references
        assert isinstance(vs.session_snapshot, dict) and vs.session_snapshot

    def test_run_duration(self, run):
        assert run.duration_s == pytest.approx(101.0)


def test_severity_ordering():
    assert Severity.INFO < Severity.SEV_3 < Severity.SEV_2 < Severity.SEV_1
    assert Severity.parse("SEV-1") == Severity.SEV_1
    assert Severity.SEV_2.label == "SEV-2"


class TestDeterministicChecks:
    def test_d1_flags_duplicate_db_call(self, run, graph):
        card = build_scorecard(run, graph, {})
        d1 = [f for f in card.flaws if f.check_id == "D1"]
        assert len(d1) == 1
        assert d1[0].node == "pmi_ddn_fetch_msp_loan_data"
        assert d1[0].severity == Severity.SEV_2

    def test_d1_ignores_control_flow(self, run, graph):
        card = build_scorecard(run, graph, {})
        d1_nodes = {f.node for f in card.flaws if f.check_id == "D1"}
        control_flow_names = {
            n.name for n in graph.nodes.values() if n.kind == "control_flow"
        }
        assert not (d1_nodes & control_flow_names)

    def test_d2_terminal_output_not_flagged(self, run, graph):
        card = build_scorecard(run, graph, {})
        dead = {f.evidence["output_key"] for f in card.flaws if f.check_id == "D2"}
        assert "pmi_ddn_final_output" not in dead

    def test_d2_router_output_not_flagged(self, run, graph):
        card = build_scorecard(run, graph, {})
        dead = {f.evidence["output_key"] for f in card.flaws if f.check_id == "D2"}
        assert "routing_decision" not in dead

    def test_d2_genuinely_unused_key_is_flagged(self, run, graph):
        card = build_scorecard(run, graph, {})
        dead = {f.evidence["output_key"] for f in card.flaws if f.check_id == "D2"}
        assert "pmi_ddn_investor_score" in dead

    def test_or3_latency_breach_flagged_with_top_contributor(self, run, graph):
        card = build_scorecard(run, graph, {"latency_budget_s": 60})
        or3 = [f for f in card.flaws if f.check_id == "O-R3"]
        assert len(or3) == 1
        assert or3[0].severity == Severity.SEV_1
        top = or3[0].evidence["top_contributors"]
        assert top[0]["agent"] == "pmi_ddn_verdict_synthesizer"

    def test_or3_not_flagged_under_budget(self, run, graph):
        card = build_scorecard(run, graph, {"latency_budget_s": 1000})
        assert not [f for f in card.flaws if f.check_id == "O-R3"]


class TestPromptChecks:
    def test_n3_seeded_undeclared_injection_flagged(self, run, graph):
        card = build_scorecard(run, graph, {})
        n3 = [f for f in card.flaws if f.check_id == "N3"]
        assert len(n3) == 1
        assert n3[0].node == "pmi_ddn_verdict_synthesizer"
        assert n3[0].evidence["session_key"] == "pmi_ddn_msp_loan_data"

    def test_n3_qa_q25527_not_flagged(self, run, graph):
        card = build_scorecard(run, graph, {})
        n3_nodes = {f.node for f in card.flaws if f.check_id == "N3"}
        assert "pmi_ddn_qa_q25527" not in n3_nodes


class TestStructuralChecks:
    def test_no_structural_drift_on_valid_run(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id == "D9"]

    def test_no_sequential_reorder_from_duplicated_child(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id == "D5"]

    def test_t_s7_flags_strict_false(self, run, graph):
        card = build_scorecard(run, graph, {})
        t_s7 = [f for f in card.flaws if f.check_id == "T-S7"]
        assert len(t_s7) == 1
        assert t_s7[0].node == "pmi_ddn_icmp_required_flag"

    def test_d5_missing_child_flagged(self):
        parent = Span(
            span_id="p", trace_id="t", parent_id=None, name="seq",
            start_ns=0, end_ns=10,
            attributes={
                "o2a.agent.name": "seq_node",
                "o2a.agent.class": "SequentialAgent",
            },
        )
        child_a = Span(
            span_id="a", trace_id="t", parent_id="p", name="a",
            start_ns=0, end_ns=1,
            attributes={"o2a.agent.name": "step_a", "o2a.agent.class": "transformation_agent"},
        )
        parent.children = [child_a]
        node = Node(name="seq_node", agent_class="SequentialAgent", sub_agent_names=["step_a", "step_b"])
        graph = _FakeGraph({"seq_node": node})
        run = Run(trace_id="t", pipeline_name="p", pipeline_hash="h", spans=[parent, child_a])
        flaws = sequential_order(run, graph, {})
        missing = [f for f in flaws if f.type == "sequential_missing"]
        assert len(missing) == 1
        assert missing[0].evidence["missing"] == ["step_b"]

    def test_d5_reorder_flagged(self):
        parent = Span(
            span_id="p", trace_id="t", parent_id=None, name="seq",
            start_ns=0, end_ns=10,
            attributes={"o2a.agent.name": "seq_node", "o2a.agent.class": "SequentialAgent"},
        )
        child_b = Span(
            span_id="b", trace_id="t", parent_id="p", name="b",
            start_ns=0, end_ns=1,
            attributes={"o2a.agent.name": "step_b", "o2a.agent.class": "transformation_agent"},
        )
        child_a = Span(
            span_id="a", trace_id="t", parent_id="p", name="a",
            start_ns=1, end_ns=2,
            attributes={"o2a.agent.name": "step_a", "o2a.agent.class": "transformation_agent"},
        )
        parent.children = [child_b, child_a]
        node = Node(name="seq_node", agent_class="SequentialAgent", sub_agent_names=["step_a", "step_b"])
        graph = _FakeGraph({"seq_node": node})
        run = Run(trace_id="t", pipeline_name="p", pipeline_hash="h", spans=[parent, child_a, child_b])
        flaws = sequential_order(run, graph, {})
        assert [f for f in flaws if f.type == "sequential_reorder"]

    def test_q_r2_flags_error_then_sibling_ran(self):
        parent = Span(
            span_id="p", trace_id="t", parent_id=None, name="seq",
            start_ns=0, end_ns=10,
            attributes={"o2a.agent.name": "seq_node", "o2a.agent.class": "SequentialAgent"},
        )
        errored = Span(
            span_id="a", trace_id="t", parent_id="p", name="a",
            start_ns=0, end_ns=1, status="ERROR",
            attributes={"o2a.agent.name": "step_a", "o2a.agent.class": "transformation_agent"},
        )
        sibling = Span(
            span_id="b", trace_id="t", parent_id="p", name="b",
            start_ns=1, end_ns=2,
            attributes={"o2a.agent.name": "step_b", "o2a.agent.class": "transformation_agent"},
        )
        parent.children = [errored, sibling]
        run = Run(trace_id="t", pipeline_name="p", pipeline_hash="h", spans=[parent, errored, sibling])
        flaws = error_swallowed(run, None, {})
        assert len(flaws) == 1
        assert flaws[0].evidence["errored_child"] == "step_a"

    def test_d6_flags_unresolved_transform_ref(self):
        node = Node(
            name="tx", agent_class="slv_transformation_agent", strict=True,
            transform={"out": {"$cond": {"if": {"left": "{{ missing_key }}"}}}},
        )
        graph = _FakeGraph({"tx": node})
        span = Span(
            span_id="s", trace_id="t", parent_id=None, name="tx",
            start_ns=0, end_ns=1,
            attributes={
                "o2a.agent.name": "tx",
                "o2a.agent.class": "slv_transformation_agent",
                "o2a.session.keys_before": '["other_key"]',
            },
        )
        run = Run(trace_id="t", pipeline_name="p", pipeline_hash="h", spans=[span], by_agent={"tx": [span]})
        flaws = transform_ref_unresolved(run, graph, {})
        assert len(flaws) == 1
        assert flaws[0].evidence["unresolved"] == ["missing_key"]
        assert flaws[0].severity == Severity.SEV_2


class TestRouterChecks:
    def test_r_s3_eq_only_router_flagged(self, run, graph):
        card = build_scorecard(run, graph, {})
        r_s3_nodes = {f.node for f in card.flaws if f.check_id == "R-S3"}
        assert "pmi_ddn_icmp_decision_router" in r_s3_nodes

    def test_r_s3_neq_router_not_flagged(self, run, graph):
        card = build_scorecard(run, graph, {})
        r_s3_nodes = {f.node for f in card.flaws if f.check_id == "R-S3"}
        assert "pmi_ddn_investor_review_router" not in r_s3_nodes

    def test_r_r1_no_misdecision_on_correct_routing(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id == "R-R1"]

    def test_r_s2_no_false_positives_on_valid_targets(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id == "R-S2"]

    def test_r_s1_no_false_positives(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id == "R-S1"]

    def test_r_s5_no_false_positives(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id == "R-S5"]

    def test_r_s7_no_false_positives(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id == "R-S7"]

    def test_r_r1_no_match_flagged(self):
        from o2a_eval.checks.router import router_misdecision

        span = Span(
            span_id="s", trace_id="t", parent_id=None, name="router",
            start_ns=0, end_ns=1,
            attributes={
                "o2a.agent.name": "router",
                "o2a.agent.class": "decision_router_agent",
                "o2a.router.evaluated_routes": '[{"target":"a","priority":10,"conditions":[],"matched":false}]',
                "o2a.router.chosen_target": None,
            },
        )
        run = Run(trace_id="t", pipeline_name="p", pipeline_hash="h", spans=[span])
        flaws = router_misdecision(run, None, {})
        assert len(flaws) == 1
        assert flaws[0].type == "router_no_match"

    def test_r_r1_degrades_without_matched_flags(self):
        from o2a_eval.checks.router import router_misdecision

        span = Span(
            span_id="s", trace_id="t", parent_id=None, name="router",
            start_ns=0, end_ns=1,
            attributes={
                "o2a.agent.name": "router",
                "o2a.agent.class": "decision_router_agent",
                "o2a.router.evaluated_routes": '[{"target":"a","priority":10,"conditions":[]}]',
                "o2a.router.chosen_target": "a",
            },
        )
        run = Run(trace_id="t", pipeline_name="p", pipeline_hash="h", spans=[span])
        flaws = router_misdecision(run, None, {})
        assert flaws == []


class TestDatabaseChecks:
    def _graph_with_query(self, query, **node_kwargs):
        defaults = dict(
            name="fetch",
            agent_class="database_agent",
            output_key="fetched",
            input_keys=["loan_number"],
            query=query,
            db_type="postgres",
            db_url="${ENV:DB_CONN}",
        )
        defaults.update(node_kwargs)
        node = Node(**defaults)
        return _FakeGraph({"fetch": node}), node

    def test_s5_no_injection_flag_with_param_binding(self):
        from o2a_eval.checks.database import sql_injection_risk

        graph, _ = self._graph_with_query("SELECT a FROM t WHERE id = :loan_number")
        assert sql_injection_risk(None, graph, {}) == []

    def test_s5_injection_flag_with_double_brace(self):
        from o2a_eval.checks.database import sql_injection_risk

        graph, _ = self._graph_with_query("SELECT a FROM t WHERE id = '{{ loan_number }}'")
        flaws = sql_injection_risk(None, graph, {})
        assert len(flaws) == 1
        assert flaws[0].severity == Severity.SEV_1

    def test_s3_no_select_star_flag_with_named_columns(self):
        from o2a_eval.checks.database import sql_select_star

        graph, _ = self._graph_with_query("SELECT a, b FROM t WHERE id = :loan_number")
        assert sql_select_star(None, graph, {}) == []

    def test_s3_flags_select_star(self):
        from o2a_eval.checks.database import sql_select_star

        graph, _ = self._graph_with_query("SELECT * FROM t WHERE id = :loan_number")
        flaws = sql_select_star(None, graph, {})
        assert len(flaws) == 1

    def test_s6_no_credentials_flag_with_env_ref(self):
        from o2a_eval.checks.database import sql_inlined_credentials

        graph, _ = self._graph_with_query("SELECT a FROM t", db_url="${ENV:DB_CONN}")
        assert sql_inlined_credentials(None, graph, {}) == []

    def test_s6_flags_inlined_credentials(self):
        from o2a_eval.checks.database import sql_inlined_credentials

        graph, _ = self._graph_with_query(
            "SELECT a FROM t", db_url="postgres://user:pass@host/db"
        )
        flaws = sql_inlined_credentials(None, graph, {})
        assert len(flaws) == 1

    def test_s8_flags_mutating_statement(self):
        from o2a_eval.checks.database import sql_mutating_statement

        graph, _ = self._graph_with_query("DELETE FROM t WHERE id = :loan_number")
        flaws = sql_mutating_statement(None, graph, {})
        assert len(flaws) == 1

    def test_s2_flags_undeclared_bound_param(self):
        from o2a_eval.checks.database import sql_param_mismatch

        graph, _ = self._graph_with_query(
            "SELECT a FROM t WHERE id = :other_param", input_keys=["loan_number"]
        )
        flaws = sql_param_mismatch(None, graph, {})
        undeclared = [f for f in flaws if f.type == "query_param_undeclared"]
        assert len(undeclared) == 1

    def test_database_checks_absent_without_sqlglot(self, monkeypatch):
        import o2a_eval.checks.database as database_mod

        monkeypatch.setattr(database_mod, "HAS_SQLGLOT", False)
        graph, _ = self._graph_with_query("SELECT a FROM t")
        assert database_mod.sql_fails_to_parse(None, graph, {}) == []
        assert database_mod.sql_unbounded_scan(None, graph, {}) == []
        assert database_mod.sql_high_complexity(None, graph, {}) == []

    def test_r1_flags_zero_rows(self):
        from o2a_eval.checks.database import zero_rows_indexed

        node = Node(name="fetch", agent_class="database_agent", output_key="rows")
        graph = _FakeGraph({"fetch": node})
        span = Span(
            span_id="s", trace_id="t", parent_id=None, name="fetch",
            start_ns=0, end_ns=1,
            attributes={
                "o2a.agent.name": "fetch",
                "o2a.agent.class": "database_agent",
                "o2a.agent.output_key": "rows",
                "o2a.db.row_count": 0,
            },
        )
        run = Run(trace_id="t", pipeline_name="p", pipeline_hash="h", spans=[span])
        flaws = zero_rows_indexed(run, graph, {})
        assert len(flaws) == 1
        assert flaws[0].severity == Severity.SEV_2

    def test_r5_flags_over_threshold(self):
        from o2a_eval.checks.database import row_count_over_threshold

        span = Span(
            span_id="s", trace_id="t", parent_id=None, name="fetch",
            start_ns=0, end_ns=1,
            attributes={
                "o2a.agent.name": "fetch",
                "o2a.agent.class": "database_agent",
                "o2a.db.row_count": 5000,
            },
        )
        run = Run(trace_id="t", pipeline_name="p", pipeline_hash="h", spans=[span])
        flaws = row_count_over_threshold(run, None, {})
        assert len(flaws) == 1


class TestPromptStructureChecks:
    def test_n2_shared_block_flagged_over_threshold(self, run, graph):
        card = build_scorecard(run, graph, {})
        n2 = [f for f in card.flaws if f.check_id == "N2"]
        assert len(n2) == 1
        assert n2[0].evidence["overlap_ratio"] > 0.25
        assert {n2[0].evidence["node_a"], n2[0].evidence["node_b"]} == {
            "pmi_ddn_qa_q25527",
            "pmi_ddn_verdict_synthesizer",
        }

    def test_n1a_no_false_positives_on_valid_refs(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id == "N1a"]

    def test_n1a_flags_unknown_ref(self):
        from o2a_eval.checks.prompts import instruction_unknown_ref

        node = Node(
            name="llm", agent_class="LlmAgent",
            instruction="Use {totally_unknown_key} to answer.",
            input_keys=[],
        )
        graph = _FakeGraph({"llm": node})
        flaws = instruction_unknown_ref(None, graph, {})
        assert len(flaws) == 1
        assert flaws[0].evidence["missing_refs"] == ["totally_unknown_key"]

    def test_n1b_flags_long_instruction(self):
        from o2a_eval.checks.prompts import instruction_too_long

        node = Node(name="llm", agent_class="LlmAgent", instruction="word " * 3000)
        graph = _FakeGraph({"llm": node})
        flaws = instruction_too_long(None, graph, {"prompt_token_threshold": 100})
        assert len(flaws) == 1

    def test_n1d_flags_missing_output_format(self):
        from o2a_eval.checks.prompts import instruction_missing_output_format

        llm = Node(
            name="llm", agent_class="LlmAgent", output_key="result",
            instruction="Just answer the question with your best judgement.",
        )
        parser = Node(name="parser", agent_class="transformation_agent", input_keys=["result"])
        graph = _FakeGraph({"llm": llm, "parser": parser})
        graph.consumers = {"result": ["parser"]}
        flaws = instruction_missing_output_format(None, graph, {})
        assert len(flaws) == 1

    def test_n1d_not_flagged_when_format_stated(self):
        from o2a_eval.checks.prompts import instruction_missing_output_format

        llm = Node(
            name="llm", agent_class="LlmAgent", output_key="result",
            instruction="Return strict JSON: {\"answer\": str}",
        )
        parser = Node(name="parser", agent_class="transformation_agent", input_keys=["result"])
        graph = _FakeGraph({"llm": llm, "parser": parser})
        graph.consumers = {"result": ["parser"]}
        assert instruction_missing_output_format(None, graph, {}) == []


class TestLifecycleChecks:
    def test_g_r1_gate_noop_flagged(self, run, graph):
        card = build_scorecard(run, graph, {})
        g_r1 = [f for f in card.flaws if f.check_id == "G-R1"]
        assert len(g_r1) == 1
        assert g_r1[0].node == "pmi_ddn_approval_gate"

    def test_no_false_positives_on_valid_run(self, run, graph):
        card = build_scorecard(run, graph, {})
        for check_id in ("G-S1", "G-R2", "O-S1", "O-S3", "Q-S1", "Q-S3"):
            assert not [f for f in card.flaws if f.check_id == check_id], check_id

    def test_g_s1_flags_unproduced_gate_input(self):
        from o2a_eval.checks.lifecycle import gate_unknown_input

        node = Node(name="gate", agent_class="agent_gate", input_keys=["nowhere"])
        graph = _FakeGraph({"gate": node})
        flaws = gate_unknown_input(None, graph, {})
        assert len(flaws) == 1
        assert flaws[0].evidence["missing"] == ["nowhere"]

    def test_o_s3_flags_unproduced_pipeline_output(self):
        from o2a_eval.checks.lifecycle import pipeline_output_unproduced

        root = Node(name="root", agent_class="resumable_orchestrator", output_key="final")
        graph = _FakeGraph({"root": root})
        flaws = pipeline_output_unproduced(None, graph, {})
        assert len(flaws) == 1

    def test_q_s1_flags_missing_node(self):
        from o2a_eval.checks.lifecycle import unresolved_reference

        parent = Node(name="parent", agent_class="SequentialAgent", sub_agent_names=["ghost"])
        ghost = Node(name="ghost", agent_class="<missing>")
        graph = _FakeGraph({"parent": parent, "ghost": ghost})
        flaws = unresolved_reference(None, graph, {})
        assert len(flaws) == 1
        assert flaws[0].evidence["referenced_by"] == ["parent"]

    def test_q_s3_flags_duplicate_child(self):
        from o2a_eval.checks.lifecycle import duplicate_sub_agent

        node = Node(name="parent", agent_class="SequentialAgent", sub_agent_names=["a", "a", "b"])
        graph = _FakeGraph({"parent": node})
        flaws = duplicate_sub_agent(None, graph, {})
        assert len(flaws) == 1
        assert flaws[0].evidence["duplicates"] == ["a"]


class TestRollup:
    def test_axis_keys(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert set(card.axes) == {
            "structural", "efficiency", "routing", "prompt_quality", "output_quality",
        }

    def test_overall_bounds(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert 0.0 <= card.overall <= 1.0

    def test_status_fail_on_sev1(self, run, graph):
        card = build_scorecard(run, graph, {"latency_budget_s": 60})
        assert card.status == "FAIL"

    def test_empty_run_static_lint_mode(self, graph):
        from o2a_eval.core.models import Run as RunModel

        empty_run = RunModel(trace_id="static", pipeline_name=graph.name, pipeline_hash=graph.pipeline_hash, spans=[])
        card = build_scorecard(empty_run, graph, {})
        assert card.max_severity < Severity.SEV_1

    def test_check_raising_produces_check_error(self, run, graph):
        from o2a_eval.core.registry import register, get_registry

        get_registry().pop("ZZ-TEST", None)

        @register("ZZ-TEST", axis="structural", stage="static")
        def _raises(run, graph, config):
            """Always raises."""
            raise RuntimeError("boom")

        card = build_scorecard(run, graph, {})
        errs = [f for f in card.flaws if f.check_id == "ZZ-TEST"]
        assert len(errs) == 1
        assert errs[0].type == "check_error"
        assert errs[0].severity == Severity.INFO
        assert card.overall > 0
        get_registry().pop("ZZ-TEST", None)


class TestScrubber:
    @pytest.mark.parametrize(
        "key",
        [
            "input_value", "output_value", "rendered_prompt", "session_snapshot",
            "query", "judge_reason", "reason", "data", "rows", "sample_shared_phrase",
        ],
    )
    def test_blocked_keys_never_persist(self, key):
        out = scrub_evidence({key: "loan 1234567890 borrower Jane Doe SSN 123-45-6789"})
        assert key not in out
        assert "1234567890" not in json.dumps(out)

    def test_unknown_key_dropped_and_reported(self):
        out = scrub_evidence({"call_count": 2, "new_field": "sensitive data"})
        assert out["call_count"] == 2
        assert "new_field" not in out
        assert "new_field" in out["_dropped_fields"]

    def test_nested_blocked_keys_removed(self):
        out = scrub_evidence({"top_contributors": [{"agent": "a", "query": "SELECT ssn"}]})
        assert "ssn" not in json.dumps(out)

    def test_full_scorecard_no_pii(self, run, graph):
        card = build_scorecard(run, graph, {"latency_budget_s": 60})
        blob = json.dumps(scorecard_to_safe_dict(card))
        for pii in ["1234567890", "Jane Doe", "ABSOLUTE RULE", "SELECT"]:
            assert pii not in blob

    def test_trace_summary_key_names_not_values(self, run):
        summary = trace_summary(run)
        blob = json.dumps(summary)
        assert "pmi_ddn_msp_loan_data" in blob
        assert "1234567890" not in blob
        assert "ABSOLUTE RULE" not in blob


class TestEphemeralStore:
    def test_memory_purge(self):
        store = EphemeralTraceStore()
        store.add({"span_id": "a", "attributes": {"secret": "loan 1234567890"}})
        assert len(store) == 1
        result = store.cleanup()
        assert result["spans_purged"] == 1
        assert len(store) == 0

    def test_cleanup_idempotent(self):
        store = EphemeralTraceStore()
        store.add({"span_id": "a"})
        store.cleanup()
        assert store.cleanup()["already_clean"] is True

    def test_add_after_cleanup_noop(self):
        store = EphemeralTraceStore()
        store.cleanup()
        store.add({"span_id": "b"})
        assert len(store) == 0

    def test_encrypted_temp_unreadable_at_rest(self, tmp_path):
        pytest.importorskip("cryptography")
        store = EphemeralTraceStore(mode="encrypted_temp", temp_dir=str(tmp_path))
        store.add({"span_id": "a", "attributes": {"v": "loan 1234567890"}})
        path = store._temp_path
        raw = path.read_bytes()
        assert b"1234567890" not in raw
        store.cleanup()

    def test_encrypted_temp_shredded_on_cleanup(self, tmp_path):
        pytest.importorskip("cryptography")
        store = EphemeralTraceStore(mode="encrypted_temp", temp_dir=str(tmp_path))
        path = store._temp_path
        store.add({"span_id": "a"})
        store.cleanup()
        assert not path.exists()

    def test_encrypted_temp_permissions(self, tmp_path):
        pytest.importorskip("cryptography")
        store = EphemeralTraceStore(mode="encrypted_temp", temp_dir=str(tmp_path))
        try:
            assert oct(store._temp_path.stat().st_mode)[-3:] == "600"
        finally:
            store.cleanup()


class TestJudgeGating:
    def test_n4_n5_absent_without_judge_config(self, run, graph):
        card = build_scorecard(run, graph, {})
        assert not [f for f in card.flaws if f.check_id in ("N4", "N5")]

    def test_local_judge_client_rejects_external_endpoint(self):
        from o2a_eval.checks.judge import JudgeEndpointError, LocalJudgeClient

        with pytest.raises(JudgeEndpointError):
            LocalJudgeClient("https://api.openai.com/v1/chat/completions", "m")

    def test_local_judge_client_accepts_localhost(self):
        from o2a_eval.checks.judge import LocalJudgeClient

        client = LocalJudgeClient("http://localhost:8080/v1/chat/completions", "m")
        assert client.calls_used == 0

    def test_local_judge_client_accepts_internal_suffix(self):
        from o2a_eval.checks.judge import LocalJudgeClient

        client = LocalJudgeClient("http://llm.corp.internal/v1/chat/completions", "m")
        assert client.exhausted is False

    def test_local_judge_client_complete_json_strips_fences(self, monkeypatch):
        from o2a_eval.checks.judge import LocalJudgeClient

        client = LocalJudgeClient("http://localhost:9000/v1", "m", max_calls=2)
        monkeypatch.setattr(client, "complete", lambda prompt: '```json\n{"a": 1}\n```')
        assert client.complete_json("hi") == {"a": 1}

    def test_local_judge_client_complete_json_fallback_regex(self, monkeypatch):
        from o2a_eval.checks.judge import LocalJudgeClient

        client = LocalJudgeClient("http://localhost:9000/v1", "m", max_calls=2)
        monkeypatch.setattr(client, "complete", lambda prompt: 'sure, here you go: {"a": 2} thanks')
        assert client.complete_json("hi") == {"a": 2}

    def test_complete_raises_when_exhausted(self):
        from o2a_eval.checks.judge import LocalJudgeClient

        client = LocalJudgeClient("http://localhost:9000/v1", "m", max_calls=0)
        assert client.exhausted is True
        with pytest.raises(RuntimeError):
            client.complete("hi")

    def test_n5_flags_low_score_and_stops(self, run, graph):
        from o2a_eval.checks.judge import judge_output_quality

        class FakeJudge:
            calls_used = 0
            max_calls = 10

            @property
            def exhausted(self):
                return self.calls_used >= self.max_calls

            def complete_json(self, prompt):
                self.calls_used += 1
                return {
                    "role_fulfillment": {"score": 2, "reason": "off task"},
                    "input_utilization": {"score": 4, "reason": "fine"},
                    "output_completeness": {"score": 4, "reason": "fine"},
                    "scope_adherence": {"score": 4, "reason": "fine"},
                    "failure_mode": "wrong_role",
                }

        flaws = judge_output_quality(run, graph, {"_judge": FakeJudge()})
        assert flaws
        assert flaws[0].severity == Severity.SEV_1

    def test_n5_judge_exception_yields_info_flaw(self, run, graph):
        from o2a_eval.checks.judge import judge_output_quality

        class ExplodingJudge:
            calls_used = 0
            max_calls = 10
            exhausted = False

            def complete_json(self, prompt):
                raise RuntimeError("judge unreachable")

        flaws = judge_output_quality(run, graph, {"_judge": ExplodingJudge()})
        assert flaws
        assert all(f.type == "judge_error" and f.severity == Severity.INFO for f in flaws)

    def test_judge_checks_suppressed_after_structural_sev1(self, run, graph):
        from o2a_eval.core.registry import register, get_registry

        get_registry().pop("ZZ-STRUCT1", None)
        get_registry().pop("ZZ-JUDGE", None)

        @register("ZZ-STRUCT1", axis="structural", stage="static")
        def _struct1(run, graph, config):
            """Fires a structural SEV-1."""
            return [Flaw(type="t", severity=Severity.SEV_1, node="x", description="d", fix=None, evidence={})]

        calls = {"n": 0}

        @register("ZZ-JUDGE", axis="output_quality", stage="runtime", needs_judge=True)
        def _judgecheck(run, graph, config):
            """A judge-gated check."""
            calls["n"] += 1
            return []

        try:
            build_scorecard(run, graph, {"_judge": object()})
            assert calls["n"] == 0
        finally:
            get_registry().pop("ZZ-STRUCT1", None)
            get_registry().pop("ZZ-JUDGE", None)


class _FakeGraph:
    """Minimal PipelineGraph stand-in for isolated check unit tests."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.root = next(iter(nodes.values()))
        self.producers = {}
        self.consumers = {}
        self.pipeline_hash = "fake"
        self.name = "fake"

    def downstream_of(self, name):
        return []
