#!/usr/bin/env python3
"""
Complete end-to-end example of kv-planner.

Demonstrates the full workflow from planning to deployment configuration.
"""

import sys
sys.path.insert(0, "/home/hesham-haroun/kv-planner/src")

from kv_planner.application import DeploymentPlanner, export


def main():
    """Run complete planning workflow."""
    print("=" * 80)
    print("kv-planner: Complete End-to-End Example")
    print("=" * 80)

    # Step 1: Create planner
    print("\n📋 Step 1: Initialize DeploymentPlanner")
    planner = DeploymentPlanner()
    print("✓ Planner ready")

    # Step 2: Create deployment plan
    print("\n📋 Step 2: Create Deployment Plan")
    print("  Model: meta-llama/Llama-3-8b-hf")
    print("  Hardware: RTX-5090")
    print("  Target: 10 requests/sec")
    print("  Goal: Balanced (cost + performance + quality)")

    plan = planner.create_plan(
        model="meta-llama/Llama-3-8b-hf",
        hardware="RTX-5090",
        target_rps=10.0,
        input_length=2048,
        output_length=512,
        optimization_goal="balanced",
        system_prompt_length=512,  # For prefix caching
    )

    print("\n✓ Plan created successfully")

    # Step 3: Display summary
    print("\n" + "=" * 80)
    print("📊 DEPLOYMENT PLAN SUMMARY")
    print("=" * 80)
    print(plan.summary)

    # Step 4: Show detailed recommendations
    print("\n" + "=" * 80)
    print("💡 DETAILED RECOMMENDATIONS")
    print("=" * 80)

    print("\n🔧 Configuration:")
    print(f"  • Precision: {plan.recommended_precision.upper()}")
    print(f"    → {plan.quantization.memory_savings_pct:.0f}% memory savings")
    print(f"    → {plan.quantization.speed_improvement:.2f}× speed improvement")
    print(f"    → {plan.quantization.quality_impact} quality impact")
    print(f"  • Batch Size: {plan.recommended_batch_size}")
    print(f"  • Prefix Caching: {'Enabled' if plan.enable_prefix_caching else 'Disabled'}")
    if plan.caching:
        print(f"    → {plan.caching.latency_reduction_pct:.0f}% latency reduction")
        print(f"    → {plan.caching.memory_savings_pct:.0f}% memory savings")

    print("\n📊 Performance:")
    print(f"  • Throughput: {plan.performance.throughput_tokens_per_sec:,.0f} tokens/sec")
    print(f"  • Latency: {plan.performance.total_latency_ms:.0f} ms")
    print(f"  • MFU: {plan.performance.mfu*100:.1f}%")
    print(f"  • MBU: {plan.performance.mbu*100:.1f}%")

    print("\n💰 Cost:")
    print(f"  • Per hour: ${plan.cost.cost_per_hour:.2f}")
    print(f"  • Per million tokens: ${plan.cost.cost_per_million_tokens:.2f}")
    print(f"  • Monthly (24/7): ${plan.cost.monthly_cost_usd:,.2f}")
    print(f"  • Annual: ${plan.cost.annual_cost_usd:,.2f}")

    print("\n💾 Memory:")
    print(f"  • Model: {plan.model_memory_gb:.2f} GB")
    print(f"  • KV Cache: {plan.kv_cache_memory_gb:.2f} GB")
    print(f"  • Total: {plan.total_memory_gb:.2f} GB / {plan.hardware.gpu_memory_gb:.0f} GB ({plan.memory_utilization_pct:.1f}%)")

    # Step 5: Export to different formats
    print("\n" + "=" * 80)
    print("📤 Step 3: Export Deployment Plan")
    print("=" * 80)

    # Export to JSON
    json_output = export.to_json(plan, indent=2)
    with open("/tmp/deployment_plan.json", "w") as f:
        f.write(json_output)
    print("\n✓ Exported to JSON: /tmp/deployment_plan.json")
    print(f"  Size: {len(json_output)} bytes")

    # Export to YAML
    yaml_output = export.to_yaml(plan)
    with open("/tmp/deployment_plan.yaml", "w") as f:
        f.write(yaml_output)
    print("✓ Exported to YAML: /tmp/deployment_plan.yaml")
    print(f"  Size: {len(yaml_output)} bytes")

    # Export to Markdown
    md_output = export.to_markdown(plan)
    with open("/tmp/deployment_plan.md", "w") as f:
        f.write(md_output)
    print("✓ Exported to Markdown: /tmp/deployment_plan.md")
    print(f"  Size: {len(md_output)} bytes")

    # Step 6: vLLM Configuration
    print("\n" + "=" * 80)
    print("⚙️  Step 4: vLLM Configuration")
    print("=" * 80)

    print("\nGenerated vLLM config (copy to your deployment):")
    print("\n```python")
    import json
    print(json.dumps(plan.vllm_config, indent=2))
    print("```")

    print("\nUsage with vLLM:")
    print("```bash")
    print(f"python -m vllm.entrypoints.openai.api_server \\")
    print(f"  --model {plan.vllm_config['model']} \\")
    print(f"  --dtype {plan.vllm_config['dtype']} \\")
    print(f"  --max-model-len {plan.vllm_config['max_model_len']} \\")
    print(f"  --max-num-seqs {plan.vllm_config['max_num_seqs']} \\")
    print(f"  --gpu-memory-utilization {plan.vllm_config['gpu_memory_utilization']} \\")
    print(f"  {'--enable-prefix-caching' if plan.vllm_config['enable_prefix_caching'] else ''}")
    print("```")

    # Step 7: Compare with alternatives
    print("\n" + "=" * 80)
    print("🔄 Step 5: Compare with Alternative GPUs")
    print("=" * 80)

    print("\nComparing RTX 5090 with alternatives...")

    alternatives = planner.compare_options(
        model="meta-llama/Llama-3-8b-hf",
        hardware_options=["RTX-5090", "RTX-4090", "RTX-3090"],
        target_rps=10.0,
        optimization_goal="balanced",
    )

    print(f"\n{'GPU':<15} {'Precision':<10} {'Batch':<8} {'$/M tok':<12} {'Throughput':<15} {'Monthly $':<12}")
    print("-" * 90)

    for p in alternatives:
        print(
            f"{p.hardware.gpu_model:<15} "
            f"{p.recommended_precision:<10} "
            f"{p.recommended_batch_size:<8} "
            f"${p.cost.cost_per_million_tokens:<11.2f} "
            f"{p.performance.throughput_tokens_per_sec:<15,.0f} "
            f"${p.cost.monthly_cost_usd:<11,.2f}"
        )

    best = alternatives[0]
    print(f"\n✨ Best value: {best.hardware.gpu_model} (${best.cost.cost_per_million_tokens:.2f}/M tokens)")

    # Step 8: Cost comparison with APIs
    print("\n" + "=" * 80)
    print("💵 Step 6: Cost Comparison vs API Providers")
    print("=" * 80)

    api_prices = {
        "Claude 3 Sonnet": 3.00,
        "GPT-4": 30.00,
        "Gemini Pro": 0.50,
        "Hosted Llama 3 8B": 0.20,
    }

    self_hosted_cost = plan.cost.cost_per_million_tokens

    print(f"\nSelf-hosted cost: ${self_hosted_cost:.2f}/M tokens\n")
    print(f"{'Provider':<25} {'API $/M':<12} {'Savings':<15} {'% Saved':<10}")
    print("-" * 70)

    for provider, api_cost in api_prices.items():
        savings = api_cost - self_hosted_cost
        savings_pct = (savings / api_cost) * 100
        print(
            f"{provider:<25} "
            f"${api_cost:<11.2f} "
            f"${savings:<14.2f} "
            f"{savings_pct:<10.1f}%"
        )

    # Final summary
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)

    print("\n📝 Next Steps:")
    print("  1. Review the deployment plan above")
    print("  2. Check exported files in /tmp/")
    print("  3. Copy vLLM configuration to your deployment")
    print("  4. Start serving with recommended settings")
    print("\n💡 Pro Tips:")
    print("  • Use spot instances to save ~30% on cloud GPUs")
    print("  • Enable prefix caching for chatbots and RAG systems")
    print("  • Monitor actual performance and adjust batch size")
    print("  • Consider FP8 for production (minimal quality loss)")

    print("\n🎉 Happy deploying!")


if __name__ == "__main__":
    main()
