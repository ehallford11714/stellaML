from stellaml import HarnessConfig, StellaNode, run_harness


def test_run_harness_executes_agents_across_path():
    cfg = HarnessConfig(
        objective="Optimize training pipeline",
        start=StellaNode(0.2, 0.2, 0.3, 0.1, 0.9),
        target=StellaNode(0.8, 0.7, 0.8, 0.7, 1.0),
        steps=3,
    )

    result = run_harness(cfg)

    assert len(result.path) == 3
    assert len(result.outcomes) == 9  # 3 agents * 3 checkpoints
    assert result.state["telemetry"] == "enabled"
    assert result.state["review"] == "alignment-check"
