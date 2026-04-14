# Implementation Plan: Agent-X (OpenWeave R&D Engine)

**Project:** Agent-X  
**Version:** 1.1.0 (Remote-First Architecture)  
**Status:** Ready for Code Generation  

---

## 1. Core Mission & Goal
**Agent-X** is an autonomous AI Research & Development engine. Its primary goal is to ingest error traces from production AI agents, surgically extract the relevant remote codebase context entirely in-memory, analyze the logic failure, and output a structured **Agent Compatibility Manifest (ACM)**. This ACM acts as a strict query parameter for downstream agents to find architectural "cures" on GitHub or ArXiv.

---

## 2. The Remote-First Codebase Retrieval Strategy
Agent-X operates strictly via APIs. It **never** uses `git clone` or local disk storage. Code is fetched on-the-fly and parsed in-memory to guarantee security, high performance, and infinite scalability.

### A. Level 1: The Porter + Surgeon (Surgical API Fetch)
* **The Porter:** Uses the client's GitHub Personal Access Token (PAT) to hit the GitHub Content API (`/repos/{owner}/{repo}/contents/{path}?ref={commit_id}`). It fetches the exact file as a Base64 string.
* **The Surgeon:** Uses `py-tree-sitter` to parse the decoded string in-memory. It maps the `lineno` from the trace to an exact AST node (function/class) and slices that logic out.

### B. Level 2: The Detective (Semantic API Escaltion)
* **The Detective:** If the Surgeon extracts a slice that references missing upstream definitions, Agent-X queries the **Greptile API**.
* **Action:** Requests a "Semantic Slice" across the repository to fetch cross-file dependencies and external function definitions.

---

## 3. Data Schemas

### Input: The Coordinate Packet (Trace Payload)
Agent-X is triggered by receiving this exact JSON structure:
```json
{
  "trace_id": "span_90210",
  "github_pat": "ghp_xxxxxxxxxxxx",
  "repository": "client-org/ai-backend",
  "commit_id": "a1b2c3d4e5...",
  "code_location": {
    "filepath": "src/agents/router.py",
    "lineno": 42,
    "function": "route_request"
  },
  "error_context": {
    "type": "RecursionError",
    "message": "Maximum call stack size exceeded"
  }
}
```

### Output: The Agent Compatibility Manifest (ACM)
Agent-X must compile its findings into this strict JSON-LD schema:
```json
{
  "@context": "https://openweave.ai/schemas/acm/v1",
  "@type": "CompatibilityManifest",
  "problem_domain": {
    "error_type": "RecursionError",
    "symptom": "Maximum call stack size exceeded",
    "failing_node": "route_request"
  },
  "codebase_context": {
    "primary_slice": "def route_request(state): ...",
    "filepath": "src/agents/router.py",
    "dependencies": ["langchain-core==0.1.52"],
    "architecture_pattern": "Sequential Routing"
  },
  "search_intent": {
    "queries": ["prevent recursion in sequential router pattern"],
    "technical_constraints": {
      "runtime": "Python 3.11",
      "required_libraries": ["langchain-core"]
    }
  }
}
```

---

## 4. Execution Workflow (Step-by-Step)

1.  **Ingestion:** Receive the Coordinate Packet.
2.  **API Fetch (Porter):** Request the raw string of `filepath` at `commit_id` via GitHub Content API.
3.  **AST Slicing (Surgeon):** Parse the string with Tree-sitter. Extract the function definition spanning `lineno`.
4.  **Dependency Mapping:** Extract `import` statements at the top of the fetched file to determine required libraries.
5.  **Reasoning Assessment (Brain):** An LLM evaluates the AST slice + error message.
    * *Check:* "Does this slice contain the origin of the bad state?"
    * *Branch:* If NO (e.g., the function just processes bad data passed into it), trigger **Level 2 (Detective)** via Greptile.
6.  **ACM Compilation (Packager):** Aggregate the local slice, any semantic detective slices, the error message, and constraints into the final ACM JSON.
7.  **Memory Flush:** Clear all fetched code strings from RAM.

---

## 5. Implementation Structure

```text
openweave-agentx/
├── agents/
│   ├── brain.py            # LLM evaluation logic (Does the slice explain the bug?)
│   └── packager.py         # Formats findings into the ACM schema
├── fetchers/
│   ├── github_porter.py    # GitHub Content API integration (PAT auth, Base64 decoding)
│   ├── tree_sitter_util.py # In-memory AST parsing and node extraction
│   └── greptile_client.py  # Greptile API integration for semantic escalation
├── schemas/
│   ├── input_packet.py     # Pydantic model for Trace Payload
│   └── acm_manifest.py     # Pydantic model for output ACM
└── main.py                 # Orchestrator: Combines Fetchers -> Brain -> Packager
```

---

## 6. Tree-Sitter Core Logic (S-Expression)
To build `tree_sitter_util.py`, codex must use this exact query pattern to find the failing block:
```lisp
(function_definition
  name: (identifier) @function.name
) @function.scope
```
*Logic Requirement:* The parser must execute this query, iterate through the matched `@function.scope` nodes, and return the exact node where `node.start_point[0] <= lineno <= node.end_point[0]`.