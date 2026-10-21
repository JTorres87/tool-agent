# Architecture

A user asks a question through the one-shot CLI (or, once built, a chat session that keeps the conversation). The agent loop runs on Claude Haiku: it gives the model the tool definitions, runs each tool call through the dispatcher, and feeds the results back until the model answers. Tools are typed: the dispatcher validates the model's input against a Pydantic model before running `list_models`, `get_model` or `estimate_cost`, which read a SQLite catalog. The setup and seed commands fill that catalog from OpenRouter's public model list (Anthropic, OpenAI and xAI models, with prices).

Every final answer goes through two checks before it's returned. The citation check (free, in code) confirms each cited tool call happened and that the chosen estimate matches its result. The verifier (Claude Sonnet) then checks the answer's claims against the cited results. A failed check sends feedback to the loop for another try; after two verifier rejections the answer is returned flagged, with the verifier's concerns.

Dashed edges are planned: the chat session is not written yet.

```mermaid
flowchart TD

subgraph group_entry["Entry &amp; Agent Loop"]
  node_cli["One-shot CLI<br/>tool-agent"]
  node_chat["Chat Session<br/>tool-agent-chat"]
  node_agent_loop["Agent Loop<br/>run_agent on Haiku"]
  node_settings["Model Settings<br/>Haiku, Sonnet verifier"]
end

subgraph group_checks["Answer Checks"]
  node_citation_check["Citation Check<br/>free, in code"]
  node_verifier["Answer Verifier<br/>Sonnet 5.5"]
end

subgraph group_tools["Typed Tools"]
  node_registry["Tool Registry &amp; Dispatch<br/>TOOLS, dispatch"]
  node_list_models["list_models<br/>tool"]
  node_get_model["get_model<br/>tool"]
  node_estimate_cost["estimate_cost<br/>tool"]
end

subgraph group_catalog["Model Catalog"]
  node_catalog_db[("SQLite Catalog<br/>data/pricing.db")]
  node_seeder["Setup &amp; Seed<br/>tool-agent-setup"]
end

subgraph group_models["Pydantic Schemas"]
  node_schemas["Tool I/O, FinalAnswer, Verdict<br/>Pydantic models"]
end

node_user(("User<br/>actor"))
node_openrouter{{"OpenRouter Models API<br/>external"}}
node_claude_api{{"Anthropic Messages API<br/>external"}}

node_user -->|"asks"| node_cli
node_user -.->|"chats"| node_chat
node_cli -->|"calls run_agent"| node_agent_loop
node_chat -.->|"runs each turn"| node_agent_loop
node_agent_loop -->|"reads"| node_settings
node_agent_loop -->|"messages.parse"| node_claude_api
node_agent_loop -->|"dispatches tool calls"| node_registry
node_agent_loop -->|"checks citations"| node_citation_check
node_agent_loop -->|"verifies answer"| node_verifier
node_verifier -->|"checks claims"| node_claude_api
node_agent_loop -->|"returns AgentResult"| node_schemas
node_registry -->|"validates input"| node_schemas
node_registry -->|"runs"| node_list_models
node_registry -->|"runs"| node_get_model
node_registry -->|"runs"| node_estimate_cost
node_estimate_cost -->|"looks up model"| node_get_model
node_list_models -->|"queries"| node_catalog_db
node_get_model -->|"reads"| node_catalog_db
node_seeder -->|"fetches models"| node_openrouter
node_seeder -->|"replaces rows"| node_catalog_db

click node_cli "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/cli.py"
click node_chat "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/chat.py"
click node_agent_loop "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/agent.py"
click node_settings "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/config.py"
click node_citation_check "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/agent.py"
click node_verifier "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/verify.py"
click node_registry "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/tools.py"
click node_list_models "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/tools.py"
click node_get_model "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/tools.py"
click node_estimate_cost "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/tools.py"
click node_catalog_db "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/db.py"
click node_seeder "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/seed.py"
click node_schemas "https://github.com/JTorres87/tool-agent/blob/main/src/tool_agent/schemas.py"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_cli,node_chat,node_agent_loop,node_settings,node_user toneBlue
class node_citation_check,node_verifier toneAmber
class node_registry,node_list_models,node_get_model,node_estimate_cost,node_openrouter,node_claude_api toneMint
class node_catalog_db,node_seeder toneRose
class node_schemas toneIndigo
```
