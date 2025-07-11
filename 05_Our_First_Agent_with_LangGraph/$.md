The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and Opportunities
https://arxiv.org/pdf/2408.13296

link: https://www.anthropic.com/engineering/built-multi-agent-research-system
🔑 Key Concepts
Multi-Agent Architecture

A lead agent (coordinator) plans the research and spawns parallel subagents to explore different aspects in tandem.
Anthropic

Subagents perform tool-assisted searches independently and return results to the lead agent.
Anthropic

Why Multi-Agent?

Ideal for dynamic, open-ended research, where tasks can evolve.

Enables parallel exploration and uses separate context windows for better throughput.

Achieved ~90% better performance over a single-agent baseline on internal research benchmarks.
Anthropic

Main Tradeoffs

Significantly higher token usage (≈15× chat consumption; ≈4× single-agent).

Best suited for high-value tasks that justify cost and complexity.

Less effective for sequential tasks with shared dependencies.
Anthropic

🛠️ Step-by-Step Build Guide
1. Define the Workflow
Lead agent receives query → plans strategy → spawns subagents.

Subagents: independently run searches, evaluate findings, and report back.

Lead consolidates info, optionally spawns more agents, then generates final output with citations.
Anthropic

2. Prompt Engineering
Simulate agent behavior to detect failure modes early.
Anthropic

Instruct lead agent how to delegate: include objectives, output formats, tools, and boundaries.

Scale delegation based on query complexity (e.g., 1 agent for simple tasks, multiple for complex).
Anthropic

3. Tool Interface Design
List available tools upfront; tie them directly to subagent needs.

Ensure clear, focused tool descriptions to prevent missteps.
Anthropic

4. Iterative Prompt Refinement
Use models (e.g. Claude) as their own prompt engineers: diagnose and suggest prompt improvements.
Anthropic

5. Search Strategy
Start with broad queries, refine based on results.
Anthropic

Use parallel searches to speed up complex tasks (lead agent spins up 3–5 subagents; each uses multiple tools).
Anthropic

6. Evaluation Setup
Begin with small evaluation sets (20 queries) to quickly measure headline improvements.

Use an LLM judge to score items like accuracy, citation quality, completeness.
Anthropic

Perform human reviews to catch nuance issues (e.g., verifying source credibility).

7. Production Reliability
Implement durable execution, checkpoints, retries, and error recovery.

Add rich observability and tracing for debugging agent decisions.
Anthropic

Use rainbow deployments to gradually introduce updates without disrupting running agents.
Anthropic

Explore asynchronous execution in future iterations to reduce bottlenecks.
Anthropic

✅ Final Takeaways
Use parallel subagents orchestrated by a lead agent to handle complex research dynamically.

Engineer prompts intentionally: clear delegation, scale rules, tool instructions, and iterative refinement.

Build for evaluation and reliability from the start: LLM-based judging, manual testing, error handling, and monitoring.


