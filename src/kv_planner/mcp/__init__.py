"""Model Context Protocol (MCP) server — kv-planner as an agent tool.

Implements the MCP 2025-11-25 stdio protocol with no external dependency —
just JSON-RPC 2.0 over stdin/stdout. An LLM agent (Claude Code, Cursor,
Cline, Continue.dev) connects, lists our tools, and calls them inline.

Exposed tools (all powered by existing kv-planner modules):

* ``plan_deployment(model, gpu, rps, input_length, output_length, goal)``
* ``recommend_models(use_case, gpu?, limit)``
* ``size_fleet(model, target_rps, p99_latency_ms, gpu_candidates?)``
* ``explain_model(slug, gpu?, use_case)``
* ``training_plan(model, method, seq_len, batch)``
* ``memory_waterfall(model, gpu, batch, input_length, output_length, precision)``
* ``speculative_decode(target, draft, method)``
* ``reasoning_plan(model, thinking_profile, precision)``
* ``carbon_estimate(throughput_tok_s, gpu, region, num_gpus)``
* ``system_info()``

Run::

    kv-planner mcp

Then point Claude Desktop / Cursor / Cline at this command in their MCP
config — kv-planner becomes a first-class tool next to `file_read`,
`bash_exec`, etc.
"""

from kv_planner.mcp.server import main

__all__ = ["main"]
