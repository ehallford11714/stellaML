from stellaml import HarnessConfig, StellaNode, run_harness


if __name__ == "__main__":
    cfg = HarnessConfig(
        objective="Train robust multimodal forecasting model",
        start=StellaNode(0.3, 0.2, 0.2, 0.1, 0.7),
        target=StellaNode(0.9, 0.8, 0.9, 0.8, 1.0),
        steps=4,
    )
    result = run_harness(cfg)
    print(f"Objective: {result.objective}")
    print(f"Path checkpoints: {len(result.path)}")
    print(f"Agent outcomes: {len(result.outcomes)}")
    print(f"Final state: {result.state}")
