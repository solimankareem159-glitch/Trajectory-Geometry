# Sustainable Multi-Agent Systems: Comprehensive Model Comparison & Architecture Guide (February 2026)
## Executive Overview
Building cost-effective multi-agent systems requires careful model selection across a spectrum of price points, intelligence levels, and agentic capabilities. The current landscape offers an unprecedented range of options, with Chinese open-source models now rivaling Western proprietary ones on many benchmarks—often at a fraction of the cost. This report provides a complete comparison of 15+ models across pricing, tool use, context windows, and agentic benchmarks, along with architectural recommendations for building tiered multi-agent systems via OpenRouter and native APIs within Google's Antigravity platform.[^1][^2]

***
## The Agentic Capability Landscape
### Key Benchmarks for Agent Evaluation
Traditional benchmarks like MMLU or HumanEval do not adequately capture a model's ability to function as an autonomous agent. The following benchmarks specifically evaluate agentic performance:[^3][^4]

- **BFCL (Berkeley Function Calling Leaderboard)**: The de facto standard for evaluating function calling accuracy across simple calls, parallel calls, multi-turn interactions, relevance detection (knowing when *not* to call a tool), and multi-step reasoning.[^3]
- **MCPMark**: Stress-tests models on realistic Model Context Protocol tasks across Notion, GitHub, Filesystem, PostgreSQL, and Playwright—averaging 16.2 execution turns and 17.4 tool calls per task.[^3]
- **Terminal-Bench**: Tests complex terminal operations, system administration, and multi-step command execution.[^5]
- **τ²-Bench**: Evaluates tool use in enterprise scenarios with real API integrations, database queries, and multi-system orchestration.[^5]
- **IFBench**: Measures instruction-following accuracy, function calling reliability, and parameter extraction precision.[^5]
### Agentic Model Rankings (January 2026)
The latest composite agentic rankings based on Terminal-Bench, τ²-Bench, and IFBench:[^5]

| Rank | Model | Quality Index | Terminal-Bench | τ²-Bench | IFBench | License |
|------|-------|--------------|---------------|----------|---------|---------|
| 1 | GPT-5.2 (xhigh) | 50.5 | 44% | 85% | 75% | Proprietary |
| 2 | Claude Opus 4.5 (high) | 49.1 | 44% | 90% | 58% | Proprietary |
| 3 | Gemini 3 Pro Preview (high) | 47.9 | 39% | 87% | 70% | Proprietary |
| 4 | GPT-5.1 (high) | 47.0 | 43% | 82% | 73% | Proprietary |
| 5 | Kimi K2.5 (Reasoning) | 46.8 | — | — | — | Open |
| 6 | Gemini 3 Flash | 45.9 | 36% | 80% | 78% | Proprietary |
| 7 | Claude 4.5 Sonnet | 42.4 | 33% | 78% | 57% | Proprietary |
| 8 | GLM-4.7 (Thinking) | 41.7 | 30% | 96% | 68% | Open (MIT) |
| 9 | GPT-5.1 Codex (high) | 41.6 | 33% | 83% | 70% | Proprietary |

Key observation: **GLM-4.7 (Thinking)** scores an astonishing **96% on τ²-Bench** (tool use), the highest of any model, open or proprietary. For pure tool-use reliability, this Chinese open-source model is the single best option available.[^5]
### BFCL Function Calling Rankings
The Berkeley Function Calling Leaderboard (October 2025) shows a different picture, emphasizing raw function calling accuracy:[^3]

| Rank | Model | BFCL Score |
|------|-------|-----------|
| 1 | GLM-4.5 (FC) | 70.85% |
| 2 | Claude Opus 4.1 | 70.36% |
| 3 | Claude Sonnet 4 | 70.29% |
| 7 | GPT-5 | 59.22% |

Chinese models (GLM family) lead the pure function calling benchmarks, followed closely by Anthropic's Claude models.[^3]
### MCPMark Real-World Tool Use
MCPMark provides the most realistic assessment of multi-step agentic workflows. These results are humbling—even the best model only achieves ~53% pass@1:[^3]

| Model | Pass@1 | Pass@4 | Avg Cost/Run | Avg Agent Time |
|-------|--------|--------|-------------|---------------|
| GPT-5 Medium | 52.6% | 68.5% | $127.46 | 478s |
| Claude Opus 4.1 | 29.9% | — | $1,165.45 | 362s |
| Claude Sonnet 4 | 28.1% | 44.9% | $252.41 | 218s |
| o3 | 25.4% | 43.3% | $113.94 | 169s |
| Qwen-3-Coder | 24.8% | 40.9% | $36.46 | 274s |

**Cost per successful task** tells the real story: Qwen-3-Coder at ~$147/successful task is actually the most economical option, while Claude Sonnet 4 costs ~$898/successful task.[^3]

***
## Complete Model Comparison
### Pricing, Context, and Capabilities Matrix
![](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/cb2295ff96cb24f2a4d3dde1a660b54f/96c3fa1b-fbf7-44f0-bd5f-6c330025c11c/3d64048d.png)
All pricing sourced from OpenRouter and native APIs as of February 2026.[^6][^7][^8][^9][^10]

| Model | Input $/M | Output $/M | Context | Tool Calling | Open Source | Origin |
|-------|-----------|------------|---------|-------------|------------|--------|
| Qwen3-235B-A22B | $0.07 | $0.10 | 262K | ✅ | ✅ | Chinese |
| MiMo-V2-Flash | Free | Free | 256K | ✅ | ✅ (MIT) | Chinese |
| DeepSeek V3.2 | $0.25 | $0.38 | 164K | ✅ | ✅ (MIT) | Chinese |
| MiniMax M2.1 | $0.30 | $1.20 | 196K | ✅ | ✅ (MIT) | Chinese |
| Gemini 2.5 Flash | $0.30 | $2.50 | 1M | ✅ | ❌ | Western |
| DeepSeek V3 | $0.30 | $1.20 | 164K | ✅ | ✅ | Chinese |
| GLM-4.7 (Thinking) | $0.40 | $1.50 | 200K | ✅ | ✅ (MIT) | Chinese |
| Kimi K2.5 | $0.60 | $2.50 | 262K | ✅ | ✅ (MIT) | Chinese |
| Gemini 2.5 Pro | $1.25 | $10.00 | 1M | ✅ | ❌ | Western |
| GPT-5 | $1.25 | $10.00 | 400K | ✅ | ❌ | Western |
| GPT-5.1 | $1.25 | $10.00 | 400K | ✅ | ❌ | Western |
| GPT-5.2 | $1.75 | $14.00 | 400K | ✅ | ❌ | Western |
| Claude Sonnet 4 | $3.00 | $15.00 | 200K | ✅ | ❌ | Western |
| Claude Sonnet 4.5 | $3.00 | $15.00 | 200K (1M avail.) | ✅ | ❌ | Western |
| Claude Opus 4.5 | $5.00 | $25.00 | 200K | ✅ | ❌ | Western |
### Chinese Models: Deep Dive
#### DeepSeek V3.2 — The Value Champion

DeepSeek V3.2, released November 2025, achieves GPT-5-level performance at approximately 1/25th the cost. It introduces DeepSeek Sparse Attention (DSA), reducing computational complexity from O(L²) to O(kL), enabling ~50% cost reduction for long-context scenarios. Key specs:[^11]

- **Architecture**: Mixture-of-Experts, MIT license, open weights on HuggingFace[^11]
- **Benchmarks**: 93.1% AIME 2025, Codeforces rating 2386, 85.0% MMLU-Pro, 83.3% LiveCodeBench[^11]
- **Tool calling**: Native support with improved post-training optimization for tool usage and agent tasks[^12]
- **Pricing**: $0.25/M input, $0.38/M output — among the cheapest capable models available[^10]
- **Context**: 164K tokens[^10]
- **Best for**: Data analytics, SQL queries, cost-sensitive agentic pipelines[^5]

#### Qwen3-235B-A22B — The Ultra-Cheap Powerhouse

Alibaba's Qwen3-235B is a 235B parameter MoE model activating only 22B parameters per forward pass. It supports seamless switching between "thinking" mode (for complex reasoning) and "non-thinking" mode (for conversational efficiency). The Qwen family surpassed 700 million downloads on HuggingFace by January 2026, making it the world's most widely used open-source AI system.[^1][^13]

- **Pricing**: As low as $0.07/M input, $0.10/M output on OpenRouter — absurdly cheap[^9]
- **Context**: 262K tokens (extends to 131K output)[^14][^9]
- **Tool calling**: Full function calling support, Hermes-style tool use[^9]
- **MCPMark**: Qwen-3-Coder achieved 24.8% pass@1 at only $36.46 per benchmark run — the cheapest run cost among top performers[^3]
- **Best for**: High-volume routing, classification, extraction, budget multi-agent backbone[^3]

#### Qwen3-Coder-480B-A35B — The Agentic Coder

Qwen also offers a 480B total parameter coding-specific MoE model with 35B active parameters, explicitly optimized for agentic coding tasks including function calling, tool use, and long-context reasoning over repositories.[^13]

#### GLM-4.7 (Thinking) — Best Open-Source Tool Use

Zhipu AI's GLM-4.7 is a 358B MoE model (32B active) that ranks #1 among open-source models on LMArena Code Arena. Its standout feature is the **96% score on τ²-Bench**, the highest tool-use score of any model tested.[^15][^5]

- **Architecture**: 358B total / 32B active MoE, MIT license[^16][^17]
- **Pricing**: $0.40/M input, $1.50/M output[^7]
- **Context**: 200K input, 128K output[^17]
- **Key features**: "Deep Thinking" mode with preserved thinking across multi-turn conversations, per-turn thinking control for cost optimization, thinks before every tool call[^18][^17]
- **Benchmarks**: 73.8% SWE-bench Verified, 84.9 LiveCodeBench V6, 96% τ²-Bench[^19][^15]
- **Best for**: Tool-heavy agentic workflows, coding agents, complex multi-step automation[^5]

#### Kimi K2.5 — Agent Swarm Architecture

Moonshot AI's Kimi K2.5 is a 1-trillion parameter open-source model introducing a revolutionary "Agent Swarm" feature that dynamically coordinates up to 100 parallel sub-agents.[^20]

- **Pricing**: $0.60/M input, $2.50–$3.00/M output — roughly 9x cheaper than Claude Opus 4.5[^21]
- **Context**: 262K tokens[^20]
- **License**: MIT[^21]
- **Key innovation**: Agent Swarm enables unprecedented task parallelization for research and engineering tasks[^20]
- **Best for**: Parallel research tasks, long-document analysis, complex multi-threaded workflows[^20]

#### MiMo-V2-Flash — Free Frontier Performance

Xiaomi's MiMo-V2-Flash is perhaps the most remarkable value proposition in the entire landscape. It's **completely free on OpenRouter** and delivers performance comparable to Claude Sonnet 4.5 at roughly 3.5% of the cost.[^22][^23]

- **Architecture**: 309B MoE, 15B active parameters, hybrid attention (5:1 SWA:GA ratio)[^24]
- **Pricing**: Free on OpenRouter[^23]
- **Context**: 256K tokens[^24]
- **Speed**: 150 output tokens/second[^22]
- **Benchmarks**: 73.4% SWE-bench Verified, 71.7% SWE-bench Multilingual (#1 open source), matches DeepSeek V3.2 on general benchmarks[^22]
- **Tool use**: τ²-Bench scores of 95.3 (Telecom), 79.5 (Retail), 66.0 (Airline)[^25]
- **Best for**: Ultra-budget agent backbone, rapid prototyping, any task where cost is the primary constraint[^25]

#### MiniMax M2.1 — Lightweight Agent Brain

MiniMax M2.1 activates only 10B parameters from a 230B MoE architecture, achieving 74% on SWE-bench Verified at ~90% cost savings compared to Claude Sonnet 4.5.[^26]

- **Pricing**: $0.30/M input, $1.20/M output[^27]
- **Context**: 196K tokens (up to 1M via MiniMax API)[^28]
- **Speed**: 56 tokens/second, faster than average[^29]
- **Best for**: Coding agents, IDE integration, lightweight agentic workflows[^27]
### Western Models: Deep Dive
#### GPT-5 / GPT-5.1 / GPT-5.2 — The MCPMark Leaders

OpenAI's GPT-5 family leads on MCPMark real-world task completion at 52.6% pass@1. GPT-5.2 tops the composite agentic rankings with a Quality Index of 50.5.[^3][^5]

- **Pricing**: $1.25/M input, $10.00/M output (GPT-5/5.1); $1.75/$14.00 (GPT-5.2)[^6]
- **Context**: 400K tokens[^6]
- **Key strength**: Best overall MCPMark performance, strong multimodal capabilities, native tool calling[^3]
- **Cost per MCPMark run**: $127.46 (cheaper than Claude at $252.41)[^3]
- **Available via**: OpenRouter, OpenAI API

#### Gemini 2.5 Flash — The Sweet Spot

Gemini 2.5 Flash represents possibly the best balance of cost, intelligence, and context window for multi-agent systems:[^30][^31]

- **Pricing**: $0.30/M input, $2.50/M output[^30]
- **Context**: 1,000,000 tokens — the largest context window at this price point[^30]
- **Tool selection score**: Perfect 1.00 on tool selection benchmarks[^31]
- **Speed/Cost scores**: 0.90 each[^31]
- **Key features**: Configurable reasoning budgets, ~24% fewer output tokens at same quality, native multi-step tool use, MoE architecture[^32][^31]
- **Available via**: Gemini API (native), OpenRouter
- **Best for**: Long-context agentic workflows, document processing, multimodal agent tasks[^31]

#### Gemini 2.5 Pro — Maximum Context Intelligence

- **Pricing**: $1.25/M input (≤200K), $2.50 (>200K); $10.00/M output (≤200K), $15.00 (>200K)[^33]
- **Context**: 1,000,000 tokens[^34]
- **Key strength**: Tops LMArena leaderboard, native Google Search and code execution integration[^3]
- **Best for**: Applications requiring multimodal reasoning and maximum context with Google ecosystem integration[^3]

#### Gemini 3 Flash — Next Generation Value

Already appearing in agentic rankings with a Quality Index of 45.9 and 78% IFBench score:[^5]
- **Pricing**: $0.50/M input, $3.00/M output[^7]
- **Near-Pro reasoning at Flash prices**, 1M token context[^7]

#### Claude Sonnet 4 / 4.5 — Premium Reasoning

Claude models excel at complex, multi-step reasoning and structured output generation:[^3]

- **Pricing**: $3.00/M input, $15.00/M output[^35][^36]
- **Context**: 200K standard, 1M available on Sonnet 4.5[^36]
- **BFCL**: 70.29% (Sonnet 4) — 3rd highest overall[^3]
- **Sonnet 4.5**: 77.2% SWE-bench Verified, 50% Terminal-Bench, context editing feature cuts token use by 84%[^36]
- **Available via**: Anthropic API, OpenRouter
- **Best for**: Complex orchestration tasks, enterprise reasoning, coding agents requiring high accuracy[^3]

#### Claude Opus 4.5 — The Heavyweight

- **Pricing**: $5.00/M input, $25.00/M output[^37]
- **Context**: 200K tokens[^37]
- **Quality Index**: 49.1 (2nd overall), 90% τ²-Bench[^5]
- **SWE-bench**: 80.9%[^37]
- **Best for**: Most complex workflows, research, situations where accuracy justifies premium cost[^5]

***
## Recommended Architecture: Tiered Multi-Agent System
![](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/cb2295ff96cb24f2a4d3dde1a660b54f/96c3fa1b-fbf7-44f0-bd5f-6c330025c11c/fcaa9177.png)
The single most impactful cost optimization strategy is a **router-first design** that escalates queries from cheap models to expensive ones only when needed. Research shows this approach can reduce inference costs by up to 72.4% compared to using a strong model baseline throughout.[^38][^39]
### Tier 1: Ultra-Cheap Layer (Classification, Routing, Simple Tasks)
**Recommended models**: Qwen3-235B ($0.07/$0.10) or MiMo-V2-Flash (Free)[^9][^23]

Use for:
- Intent classification and query routing
- Simple Q&A, summarization, extraction
- Data formatting and transformation
- Initial planning/decomposition of complex tasks
- High-volume batch processing

**Cost impact**: At $0.07/M input tokens, processing 1 million routing decisions costs approximately $0.07 — essentially free.[^9]
### Tier 2: Balanced Layer (Tool Use, Medium Complexity)
**Recommended models**: DeepSeek V3.2 ($0.25/$0.38), Gemini 2.5 Flash ($0.30/$2.50), or GLM-4.7 ($0.40/$1.50)[^7][^30][^10]

Use for:
- Standard tool calling and function execution
- Code generation and debugging
- Multi-step workflows with moderate complexity
- Document analysis with long context (Gemini 2.5 Flash for 1M context)
- Database queries and data pipeline tasks

**Model selection within Tier 2**:
- **Need massive context?** → Gemini 2.5 Flash (1M tokens at $0.30 input)[^30]
- **Need best tool use reliability?** → GLM-4.7 (96% τ²-Bench)[^5]
- **Need cheapest capable option?** → DeepSeek V3.2 ($0.25/$0.38)[^10]
### Tier 3: Frontier Layer (Complex Reasoning, Critical Tasks)
**Recommended models**: GPT-5/5.1 ($1.25/$10.00), Claude Sonnet 4.5 ($3.00/$15.00), or Gemini 2.5 Pro ($1.25/$10.00)[^6][^33][^36]

Use for:
- Complex multi-step agentic workflows requiring high reliability
- Critical business logic where errors are costly
- Multi-system orchestration (MCPMark-style tasks)
- Research synthesis requiring deep reasoning
- Tasks that failed at Tier 2

**Model selection within Tier 3**:
- **Best overall MCPMark success?** → GPT-5 (52.6% pass@1)[^3]
- **Best reasoning depth?** → Claude Sonnet 4.5 or Opus 4.5[^5][^3]
- **Best Google ecosystem integration?** → Gemini 2.5 Pro[^3]
### Implementation via OpenRouter
OpenRouter provides a unified API compatible with all these models, making it ideal for building tiered systems within Antigravity:[^40]

- **Model catalog**: 400+ models with sortable pricing and availability[^40]
- **Auto router**: Automatically picks the "best" option based on your criteria[^40]
- **`:floor` shortcut**: Routes to the lowest-price provider for a given model[^40]
- **`:nitro` shortcut**: Routes to the fastest-throughput provider[^40]
- **BYOK support**: Bring your own API keys for direct provider access[^40]

For models where OpenRouter pricing is suboptimal, use native APIs:
- **Gemini API**: Best for Gemini 2.5 Flash/Pro with native tool calling and context caching[^41]
- **Claude API**: Best for Claude Sonnet 4.5 with prompt caching (can reduce costs significantly)[^42]
- **DeepSeek API**: Cache hits reduce token cost from $0.28 to $0.028 per 1M tokens — a 10x reduction for repeated prompts[^43]

***
## Specialized Agent Configurations
### Configuration A: Maximum Budget Savings
Total estimated cost per 1M agent decisions: **~$0.20**

| Agent Role | Model | Cost ($/M input) |
|-----------|-------|-------------------|
| Router | Qwen3-235B | $0.07 |
| Worker (simple) | MiMo-V2-Flash | Free |
| Worker (tool use) | DeepSeek V3.2 | $0.25 |
| Reviewer/QA | GLM-4.7 | $0.40 |
### Configuration B: Balanced Performance
Total estimated cost per 1M agent decisions: **~$2.00**

| Agent Role | Model | Cost ($/M input) |
|-----------|-------|-------------------|
| Router | Qwen3-235B | $0.07 |
| Worker (general) | Gemini 2.5 Flash | $0.30 |
| Worker (coding) | DeepSeek V3.2 | $0.25 |
| Orchestrator | GPT-5 | $1.25 |
### Configuration C: Maximum Capability
Total estimated cost per 1M agent decisions: **~$6.00**

| Agent Role | Model | Cost ($/M input) |
|-----------|-------|-------------------|
| Router | Gemini 2.5 Flash | $0.30 |
| Worker (tool use) | GPT-5.1 | $1.25 |
| Worker (reasoning) | Claude Sonnet 4.5 | $3.00 |
| Orchestrator | GPT-5.2 | $1.75 |

***
## Cost Optimization Strategies
### 1. Prompt Caching
Several providers offer dramatic cost reductions for cached prompts:[^38][^43]
- **DeepSeek**: Cache hits cost $0.028/M vs $0.28/M (10x savings)[^43]
- **Anthropic**: Prompt caching available on Claude models, reducing repeated context costs[^42]
- **Gemini**: Context caching available on 2.5 Pro and Flash[^33]
### 2. Configurable Reasoning Budgets
Models like Gemini 2.5 Flash and MiMo-V2-Flash support toggling "thinking" on/off:[^32][^25]
- Disable thinking for simple tasks → lower cost and latency
- Enable thinking only for complex tasks → higher quality when needed
### 3. Output Token Optimization
Output tokens are typically 3–10x more expensive than input tokens. Strategies include:
- Use concise system prompts that instruct models to be brief
- Gemini 2.5 Flash produces ~24% fewer output tokens at the same quality after recent updates[^32]
- MiMo-V2-Flash produces ~50% fewer output tokens in Flash-Lite mode[^32]
### 4. Distillation and Task-Specific SLMs
For common subtasks (routing, extraction, classification), distill knowledge from large models into task-specific small models. This reduces per-call cost and latency while preserving accuracy on narrow tasks.[^38]
### 5. Progressive Tool Discovery
Rather than overwhelming agents with all available tools at once, use progressive discovery to guide agents through tools step-by-step. This approach achieved nearly 2x the success rate on GitHub tasks in MCPMark testing.[^3]

***
## Context Window Comparison
For agents handling long documents or extended multi-turn conversations, context window size is critical:

| Context Tier | Models | Context Size |
|-------------|--------|-------------|
| Ultra-Long | Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 3 Flash | 1,000,000 tokens |
| Very Long | GPT-5/5.1/5.2 | 400,000 tokens |
| Long | Qwen3-235B, Kimi K2.5 | 262,000 tokens |
| Long | MiMo-V2-Flash | 256,000 tokens |
| Standard | Claude Sonnet 4/4.5 (standard), GLM-4.7 | 200,000 tokens |
| Standard | MiniMax M2.1 | 196,000 tokens |
| Standard | DeepSeek V3/V3.2 | 164,000 tokens |

**Note**: Claude Sonnet 4.5 supports a 1M context window option, but with higher per-token pricing for requests exceeding 200K input tokens.[^42]

For multi-agent systems where individual agents handle bounded subtasks, 164K–262K context is typically sufficient. The 1M context of Gemini 2.5 Flash becomes valuable for agents that need to process entire codebases or lengthy document sets in a single pass.[^30]

***
## Practical Implementation Notes for Antigravity
### API Routing Strategy
Since you're building within Google's Antigravity and supplying your own API keys:

1. **Primary route via OpenRouter**: Use OpenRouter as the default gateway for all models. It provides a unified OpenAI-compatible API, meaning you write one integration and access 400+ models.[^40]

2. **Direct Gemini API for Google models**: For Gemini 2.5 Flash/Pro, the native Gemini API offers better tool calling integration, context caching, and potentially lower latency for Google-native features.[^41]

3. **Direct Claude API for Anthropic models**: When using Claude Sonnet 4.5 for complex reasoning tasks, the native API gives access to prompt caching and the extended 1M context option.[^42]

4. **Direct DeepSeek API for cache-heavy workloads**: If your agents repeatedly process similar prompts, DeepSeek's 10x cache discount makes the direct API significantly cheaper.[^43]
### Recommended Starting Configuration
For an initial multi-agent system balancing cost and capability:

- **Router agent**: Qwen3-235B via OpenRouter ($0.07/M) — classifies incoming tasks by complexity
- **General worker**: Gemini 2.5 Flash via Gemini API ($0.30/$2.50) — handles 70% of tasks with 1M context
- **Tool specialist**: GLM-4.7 via OpenRouter ($0.40/$1.50) — handles tool-heavy tasks (96% τ²-Bench)
- **Escalation/QA**: Claude Sonnet 4.5 via Claude API ($3/$15) — handles only the hardest 5–10% of tasks

This configuration should handle most workloads at an average effective cost well under $1/M tokens, while maintaining frontier-level capability for the tasks that need it.

***
## Risk Considerations
While security was noted as a lower priority, a few practical risks to be aware of:

- **API reliability**: Chinese model APIs (DeepSeek, Qwen direct) have historically shown lower uptime (DeepSeek at ~81.7% via some providers). OpenRouter mitigates this with provider fallback routing.[^10]
- **Regulatory**: Chinese models may be subject to different regulatory frameworks that could affect availability.[^20]
- **Documentation**: Chinese models generally have less extensive English documentation than Western alternatives.[^3]
- **Rate limits**: Free tiers (MiMo-V2-Flash) may have rate limits that affect production workloads at scale.[^23]

***
## Conclusion and Recommendations
The optimal multi-agent strategy in February 2026 is unambiguously a **tiered architecture** that routes tasks to the cheapest capable model. Chinese open-source models have fundamentally changed the economics:

- **For routing/classification**: Qwen3-235B at $0.07/M is essentially free[^9]
- **For tool use**: GLM-4.7 at $0.40/M achieves 96% τ²-Bench — better than any proprietary model[^5]
- **For general work**: Gemini 2.5 Flash at $0.30/M input with 1M context is the sweet spot[^30]
- **For complex reasoning**: GPT-5 at $1.25/M has the best MCPMark pass@1 rate[^3]
- **For free prototyping**: MiMo-V2-Flash matches Sonnet 4.5 at zero cost[^23]

The combination of these models in a well-designed routing architecture can achieve 70%+ cost reduction compared to using a single frontier model, while maintaining equivalent or superior task completion rates.[^39]

---

## References

1. [[News] Chinese AI Models Reportedly Hit ~15% Global ...](https://www.trendforce.com/news/2026/01/26/news-chinese-ai-models-reportedly-hit-15-global-share-in-nov-2025-fueled-by-deepseek-open-source-push/) - R1, released by DeepSeek in January 2025, stunned the world by achieving high performance at low cos...

2. [Key Milestones of China in AI of 2025 - AI Supremacy](https://www.ai-supremacy.com/p/milestones-of-china-in-ai-of-2025-deepseek-qwen) - 📖 Top 10 China AI Stories in 2025: A Year-End Review 🐣

3. [Function Calling and Agentic AI in 2025: What the Latest ...](https://www.klavis.ai/blog/function-calling-and-agentic-ai-in-2025-what-the-latest-benchmarks-tell-us-about-model-performance) - A comprehensive analysis of function calling benchmarks like BFCL and MCPMark, revealing how today's...

4. [Best AI Agent Evaluation Benchmarks: 2025 Complete Guide](https://o-mega.ai/articles/the-best-ai-agent-evals-and-benchmarks-full-2025-guide) - Compare top AI agent benchmarks and evaluation frameworks for 2025. In-depth analysis of web, OS, an...

5. [Best Agentic AI Models January 2026 Rankings - WhatLLM.org](https://whatllm.org/blog/best-agentic-models-january-2026) - Definitive ranking of AI models for building autonomous agents and tool use. Based on Terminal-Bench...

6. [LLM API Pricing 2026 - Compare 300+ AI Model Costs](https://pricepertoken.com) - Free LLM API pricing comparison. Compare GPT-5, Claude, Gemini & DeepSeek costs instantly. Updated d...

7. [OpenRouter Models Ranked: 15 Best for Coding, Free & ...](https://www.teamday.ai/blog/top-ai-models-openrouter-2026) - Claude Sonnet 4. $5. Total cost: ~$7 vs $50-100+ using Claude Opus throughout. Same quality. 90% che...

8. [DeepSeek V3 API Pricing 2026](https://pricepertoken.com/pricing-page/model/deepseek-deepseek-chat) - DeepSeek V3 pricing: $0.30/M input. Compare with 10 similar models, see benchmarks, and find the che...

9. [Qwen3 235 b A22 b 2507 Pricing & Specs | AI Models | CloudPrice](https://cloudprice.net/models/openrouter%2Fqwen%2Fqwen3-235b-a22b-2507) - Compare Qwen3 235 b A22 b 2507 AI model pricing, specifications, and capabilities. View input/output...

10. [DeepSeek Deepseek-V3.2 Pricing (Updated 2025)](https://pricepertoken.com/pricing-page/model/deepseek-deepseek-v3.2) - DeepSeek V3.2 pricing: $0.25/M input. Compare with 10 similar models, see benchmarks, and find the c...

11. [DeepSeek releases V3.2 & V3.2 Speciale](https://datanorth.ai/news/deepseek-releases-v3-2-v3-2-speciale) - Open-source DeepSeek V3.2 delivers frontier AI performance 25× cheaper than alternatives. First mode...

12. [DeepSeek-V3.1 API by DEEPSEEK - Competitive Pricing](https://www.atlascloud.ai/models/deepseek-ai/DeepSeek-V3.1) - DeepSeek-V3.1 API - competitive pricing, transparent rates. Starting from $0.3/1M tokens. Unified AP...

13. [Qwen](https://openrouter.ai/qwen) - It natively handles 32K token contexts and can extend to 131K tokens using YaRN-based scaling. Qwen3...

14. [Qwen3 235B A22B Instruct 2507 - Cracked AI Engineering](https://www.crackedaiengineering.com/ai-models/openrouter-qwen-qwen3-235b-a22b-07-25) - OpenRouter's Qwen3 235B A22B Instruct 2507 AI model offers developers a powerful tool with a large c...

15. [GLM-4.7 API](https://www.together.ai/models/glm-4-7) - 200K context reasoning model ranking #1 open-source on LMArena Code Arena, with 73.8% SWE-bench Veri...

16. [GLM-4.7 - Pricing, Context Window Size, and Benchmark Data](https://automatio.ai/models/glm-4-7) - GLM-4.7 by Zhipu AI is a flagship 358B MoE model featuring a 200K context window, elite 73.8% SWE-be...

17. [GLM 4.7 API - Competitive Pricing - Unified API Access - Atlas Cloud](https://www.atlascloud.ai/models/zai-org/glm-4.7) - GLM 4.7 API - competitive pricing, transparent rates. Starting from $0.52/1M tokens. Unified API acc...

18. [GLM-4.7 - OpenLM.ai](https://openlm.ai/glm-4.7/) - The GLM-4.x series models are foundation models designed for intelligent agents. GLM-4.7 has 355 bil...

19. [GLM‑4.7: The Open‑Source Model That Sets New Benchmarks](https://blog.meetneura.ai/glm-4-7-open-source-model/) - This simple example shows how GLM‑4.7 can be used to build a tool‑aware assistant. You can expand th...

20. [Kimi K2.5 - Pricing, Context Window Size, and Benchmark Data](https://automatio.ai/models/kimi-k2-5) - Discover Moonshot AI's Kimi K2.5, a 1T-parameter open-source agentic model featuring native multimod...

21. [Kimi K2.5: Everything We Know About Moonshot's Visual ...](https://wavespeed.ai/blog/posts/kimi-k2-5-everything-we-know-about-moonshots-visual-agentic-model) - Kimi K2.5 is Moonshot AI's open-source 1T parameter model with Agent Swarm technology, 256K context,...

22. [XiaomiMiMo](https://x.com/XiaomiMiMo/status/2000929154670157939)

23. [Apps Using Xiaomi: MiMo-V2-Flash (free)](https://openrouter.ai/xiaomi/mimo-v2-flash:free/apps) - See apps that are using Xiaomi: MiMo-V2-Flash - MiMo-V2-Flash is an open-source foundation language ...

24. [XiaomiMiMo/MiMo-V2-Flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash) - MiMo-V2-Flash is a Mixture-of-Experts (MoE) language model with 309B total parameters and 15B active...

25. [MiMo-V2-Flash | Xiaomi](https://mimo.xiaomi.com/mimo-v2-flash)

26. [MiniMax M2.1 API Text by MINIMAX - Competitive Pricing - Atlas Cloud](https://www.atlascloud.ai/models/minimaxai/minimax-m2.1) - MiniMax M2.1 API - competitive pricing, transparent rates. Starting from $0.3/1M tokens. Unified API...

27. [MiniMax M2.1 by MiniMax - AI Model Details - LLMBase](https://llmbase.ai/models/minimax/minimax-m2.1/) - MiniMax-M2.1 is a lightweight, state-of-the-art large language model optimized for coding, agentic w...

28. [MiniMax M2.1: Pricing, Context Window, Benchmarks, and More](https://llm-stats.com/models/minimax-m2.1) - MiniMax M2.1 is an enhanced large language model focused on multi-language programming and real-worl...

29. [MiniMax-M2.1 - Intelligence, Performance & Price Analysis](https://artificialanalysis.ai/models/minimax-m2-1) - MiniMax-M2.1 is amongst the leading models in intelligence and reasonably priced when comparing to o...

30. [Google Gemini-2.5-Flash Pricing (Updated 2025)](https://pricepertoken.com/pricing-page/model/google-gemini-2.5-flash) - Gemini 2.5 Flash pricing: $0.30/M input. Compare with 10 similar models, see benchmarks, and find th...

31. [Gemini-2.5-flash Overview](https://galileo.ai/model-hub/gemini-2-5-flash-overview) - Explore gemini-2.5-flash performance benchmarks, industry-specific capabilities, and evaluation metr...

32. [Gemini 2.5 Flash: Improved Performance and Cost](https://www.linkedin.com/posts/vladislav-krivonos-05b45b11b_gemini25-genai-agenticai-activity-7380913021516169216-FtAz) - ⚡ Gemini 2.5 Flash just got sharper (and cheaper to run) Google’s new gemini-2.5-flash-preview-09-20...

33. [Gemini Pricing in 2026 for Individuals, Orgs & Developers](https://www.finout.io/blog/gemini-pricing-in-2026) - Model. Input Price. Output Price. Context Caching. Grounding (Google Search) ; 2.5 Pro. $1.25 (≤200k...

34. [Gemini 2.5 Pro: Pricing, Context Window, Benchmarks, and ...](https://llm-stats.com/models/gemini-2.5-pro) - Pricing starts at $1.25 per million input tokens and $10.00 per million output tokens. The model sup...

35. [Claude Sonnet 4: Pricing, Context Window, Benchmarks, and More](https://llm-stats.com/models/claude-sonnet-4-20250514) - Claude Sonnet 4, part of the Claude 4 family, is a significant upgrade to Claude Sonnet 3.7. It exce...

36. [Claude Sonnet 4.5: Features, Benchmarks & Pricing (2026)](https://www.leanware.co/insights/claude-sonnet-4-5-overview) - TLDR: Claude Sonnet 4.5 scores 77.2% on SWE-bench Verified (82.0% with parallel compute), 50.0% on T...

37. [Claude Sonnet 4.5 - Pricing, Context Window Size, and Benchmark ...](https://automatio.ai/models/claude-sonnet-4-5) - Anthropic's Claude Sonnet 4.5 delivers world-leading coding (77.2% SWE-bench) and a 200K context win...

38. [AI Agent Cost Optimization 2025: 8 Practical Steps to Shift ...](https://www.linkedin.com/pulse/ai-agent-cost-optimization-2025-8-practical-steps-t3ppe) - AI Agent Cost Optimization 2025: 8 Practical Steps to Shift to Make Costs Predictable.

39. [CASTER: Breaking the Cost-Performance Barrier in Multi- ...](https://arxiv.org/html/2601.19793v1) - A comprehensive review of these LLM-based multi-agent systems and their applications in software eng...

40. [OpenRouter Review 2025: Multi-Model LLM Gateway ...](https://skywork.ai/blog/openrouter-review-2025/) - Independent 2025 review: OpenRouter’s OpenAI-compatible API, routing options, BYOK, pricing, privacy...

41. [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing) - [*] Image output is priced at $30 per 1,000,000 tokens. Output images up to 1024x1024px consume 1290...

42. [Pricing - Claude Docs](https://platform.claude.com/docs/en/about-claude/pricing) - Long context pricing. When using Claude Opus 4.6, Sonnet 4.5, or Sonnet 4 with the 1M token context ...

43. [DeepSeek Pricing Guide 2025 for Australian Businesses](https://wise.com/au/blog/deepseek-pricing) - Understand DeepSeek pricing for 2025. Compare plans, costs and learn how Wise Business helps Austral...

