#!/usr/bin/env python3
"""RISC-V ISA Chatbot - HuggingFace Spaces App
Data from RISC-V Unified Database (UDB): https://github.com/riscv-software-src/riscv-unified-db
"""

import base64
import json
import os
import time
import urllib.request
import urllib.parse
from collections import defaultdict
from pathlib import Path

import anthropic
import gradio as gr
import yaml

DATA_DIR = Path(__file__).parent / "data"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Rate limiting: 20 questions per user (by session)
MAX_QUESTIONS_PER_USER = 20
user_usage = defaultdict(lambda: {"count": 0, "first_use": time.time()})

# Available configs - dynamically detected
AVAILABLE_CONFIGS = []
DEFAULT_CONFIG = "RV64"

# Per-config data indices
_config_data = {}


def load_yaml(path):
    """Load and cache a YAML file."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except:
        return {}


def build_config_index(config_name):
    """Build indices for a single config."""
    config_lower = config_name.lower()
    config_dir = DATA_DIR / config_lower

    if not config_dir.exists():
        return None

    inst_index = []
    csr_index = []
    ext_index = []
    yaml_cache = {}

    # Index instructions
    for yf in (config_dir / "inst").rglob("*.yaml") if (config_dir / "inst").exists() else []:
        try:
            d = load_yaml(yf)
            yaml_cache[str(yf)] = d
            if d.get("name"):
                db = d.get("definedBy", [])
                if isinstance(db, str): db = [db]
                elif isinstance(db, dict): db = db.get("anyOf", db.get("allOf", []))
                inst_index.append({
                    "path": str(yf),
                    "name": d["name"],
                    "long_name": d.get("long_name", ""),
                    "assembly": d.get("assembly", ""),
                    "definedBy": db
                })
        except: pass

    # Index CSRs
    for yf in (config_dir / "csr").rglob("*.yaml") if (config_dir / "csr").exists() else []:
        try:
            d = load_yaml(yf)
            yaml_cache[str(yf)] = d
            if d.get("name"):
                db = d.get("definedBy", [])
                if isinstance(db, str): db = [db]
                csr_index.append({
                    "path": str(yf),
                    "name": d["name"],
                    "long_name": d.get("long_name", ""),
                    "address": d.get("address"),
                    "definedBy": db
                })
        except: pass

    # Index extensions
    for yf in (config_dir / "ext").rglob("*.yaml") if (config_dir / "ext").exists() else []:
        try:
            d = load_yaml(yf)
            yaml_cache[str(yf)] = d
            if d.get("name") and d.get("kind") == "extension":
                ext_index.append({
                    "path": str(yf),
                    "name": d["name"],
                    "long_name": d.get("long_name", ""),
                    "description": d.get("description", "")[:200]
                })
        except: pass

    return {
        "inst_index": inst_index,
        "csr_index": csr_index,
        "ext_index": ext_index,
        "yaml_cache": yaml_cache
    }


def detect_configs():
    """Detect available configs from data directory."""
    global AVAILABLE_CONFIGS, DEFAULT_CONFIG

    if not DATA_DIR.exists():
        return

    for subdir in DATA_DIR.iterdir():
        if subdir.is_dir() and (subdir / "inst").exists():
            config_name = subdir.name.upper()
            if config_name not in AVAILABLE_CONFIGS:
                AVAILABLE_CONFIGS.append(config_name)

    # Sort with RV64 first
    AVAILABLE_CONFIGS.sort(key=lambda x: (0 if x == "RV64" else 1, x))

    if AVAILABLE_CONFIGS:
        DEFAULT_CONFIG = AVAILABLE_CONFIGS[0]


def build_all_indices():
    """Build indices for all available configs."""
    global _config_data

    detect_configs()

    for config in AVAILABLE_CONFIGS:
        data = build_config_index(config)
        if data:
            _config_data[config] = data
            print(f"  {config}: {len(data['inst_index'])} instructions, {len(data['csr_index'])} CSRs, {len(data['ext_index'])} extensions")


def get_config_data(config):
    """Get data for a specific config."""
    return _config_data.get(config, _config_data.get(DEFAULT_CONFIG, {}))


# Tool functions that use config context
def search_instructions(config, term="", extension="", limit=20):
    data = get_config_data(config)
    results = []
    for i in data.get("inst_index", []):
        if term and term.lower() not in i["name"].lower() and term.lower() not in i.get("long_name", "").lower():
            continue
        if extension and extension not in i.get("definedBy", []):
            continue
        results.append({"name": i["name"], "long_name": i["long_name"], "assembly": i["assembly"], "definedBy": i["definedBy"]})
        if len(results) >= limit:
            break
    return {"count": len(results), "instructions": results}


def search_csrs(config, term="", extension="", limit=20):
    data = get_config_data(config)
    results = []
    for c in data.get("csr_index", []):
        if term and term.lower() not in c["name"].lower() and term.lower() not in c.get("long_name", "").lower():
            continue
        if extension and extension not in c.get("definedBy", []):
            continue
        results.append({"name": c["name"], "long_name": c["long_name"], "address": c["address"]})
        if len(results) >= limit:
            break
    return {"count": len(results), "csrs": results}


def list_extensions(config):
    data = get_config_data(config)
    ext_index = data.get("ext_index", [])
    return {"count": len(ext_index), "extensions": [{"name": e["name"], "long_name": e["long_name"]} for e in ext_index]}


def get_extension_details(config, name):
    data = get_config_data(config)
    ext_index = data.get("ext_index", [])
    inst_index = data.get("inst_index", [])
    csr_index = data.get("csr_index", [])

    ext = next((e for e in ext_index if e["name"] == name), None)
    if not ext: return {"error": f"Extension '{name}' not found"}
    insts = [{"name": i["name"], "assembly": i["assembly"]} for i in inst_index if name in i.get("definedBy", [])][:30]
    csrs = [{"name": c["name"], "address": c["address"]} for c in csr_index if name in c.get("definedBy", [])][:20]
    return {"extension": ext, "instructions": {"count": len(insts), "items": insts}, "csrs": {"count": len(csrs), "items": csrs}}


def get_instruction_details(config, name):
    data = get_config_data(config)
    inst_index = data.get("inst_index", [])
    yaml_cache = data.get("yaml_cache", {})

    inst = next((i for i in inst_index if i["name"] == name), None)
    if not inst: return {"error": f"Instruction '{name}' not found"}
    return {"instruction": yaml_cache.get(inst["path"], load_yaml(inst["path"]))}


def get_csr_details(config, name):
    data = get_config_data(config)
    csr_index = data.get("csr_index", [])
    yaml_cache = data.get("yaml_cache", {})

    csr = next((c for c in csr_index if c["name"] == name), None)
    if not csr: return {"error": f"CSR '{name}' not found"}
    return {"csr": yaml_cache.get(csr["path"], load_yaml(csr["path"]))}


def get_stats(config):
    data = get_config_data(config)
    return {
        "config": config,
        "instructions": len(data.get("inst_index", [])),
        "csrs": len(data.get("csr_index", [])),
        "extensions": len(data.get("ext_index", []))
    }


def search_isa_manual(query):
    """Search the official RISC-V ISA Manual on GitHub."""
    ISA_MANUAL_REPO = "riscv/riscv-isa-manual"

    try:
        # Use GitHub code search API (no auth needed for public repos)
        encoded_query = urllib.parse.quote(f"{query} repo:{ISA_MANUAL_REPO}")
        url = f"https://api.github.com/search/code?q={encoded_query}&per_page=5"

        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "RISC-V-ISA-Chatbot"
        })

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        if not data.get("items"):
            return {
                "source": "RISC-V ISA Manual (GitHub)",
                "found": False,
                "message": f"No results found for '{query}' in the official ISA Manual."
            }

        results = []
        for item in data["items"][:5]:
            # Fetch file content snippet
            file_url = item.get("url")
            if file_url:
                try:
                    req2 = urllib.request.Request(file_url, headers={
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "RISC-V-ISA-Chatbot"
                    })
                    with urllib.request.urlopen(req2, timeout=5) as resp:
                        file_data = json.loads(resp.read().decode())
                        # Decode base64 content and get relevant snippet
                        content = base64.b64decode(file_data.get("content", "")).decode("utf-8", errors="ignore")
                        # Find relevant lines containing the query
                        lines = content.split("\n")
                        relevant_lines = []
                        for i, line in enumerate(lines):
                            if query.lower() in line.lower():
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                relevant_lines.extend(lines[start:end])
                                if len(relevant_lines) > 15:
                                    break
                        snippet = "\n".join(relevant_lines[:15]) if relevant_lines else content[:500]
                except:
                    snippet = ""

            results.append({
                "file": item.get("path", ""),
                "url": item.get("html_url", ""),
                "snippet": snippet[:800] if snippet else ""
            })

        return {
            "source": "RISC-V ISA Manual (GitHub)",
            "repo_url": f"https://github.com/{ISA_MANUAL_REPO}",
            "found": True,
            "count": len(results),
            "results": results
        }

    except urllib.error.HTTPError as e:
        if e.code == 403:
            return {"source": "RISC-V ISA Manual", "error": "GitHub API rate limit reached. Try again later."}
        return {"source": "RISC-V ISA Manual", "error": f"GitHub API error: {e.code}"}
    except Exception as e:
        return {"source": "RISC-V ISA Manual", "error": str(e)}


# Local database tools (always available)
LOCAL_TOOLS = [
    {"name": "search_instructions", "description": "Search RISC-V instructions by name or filter by extension",
     "input_schema": {"type": "object", "properties": {"term": {"type": "string", "description": "Search term"}, "extension": {"type": "string", "description": "Filter by extension (e.g. M, A, F, V)"}}}},
    {"name": "search_csrs", "description": "Search RISC-V CSRs by name or extension",
     "input_schema": {"type": "object", "properties": {"term": {"type": "string"}, "extension": {"type": "string"}}}},
    {"name": "list_extensions", "description": "List all RISC-V extensions",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_extension_details", "description": "Get detailed info about a RISC-V extension including its instructions and CSRs",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Extension name (e.g. M, A, V, Zba)"}}, "required": ["name"]}},
    {"name": "get_instruction_details", "description": "Get complete details about a specific instruction including encoding and operation",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Instruction name (e.g. add, mul, lw)"}}, "required": ["name"]}},
    {"name": "get_csr_details", "description": "Get complete details about a specific CSR including fields",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "CSR name (e.g. mstatus, mcause)"}}, "required": ["name"]}},
    {"name": "get_stats", "description": "Get database statistics",
     "input_schema": {"type": "object", "properties": {}}},
]

# External tool (optional, requires local search first)
EXTERNAL_TOOL = {"name": "search_isa_manual", "description": "Search the official RISC-V ISA Manual on GitHub for information not in the local database. Use this ONLY AFTER searching local tools first. Results are from an EXTERNAL source and should be clearly attributed.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query (e.g. 'fence.i rationale', 'memory ordering', 'trap handling')"}}, "required": ["query"]}}

LOCAL_TOOL_NAMES = {t["name"] for t in LOCAL_TOOLS}

TOOL_FNS = {
    "search_instructions": lambda config, **x: search_instructions(config, **x),
    "search_csrs": lambda config, **x: search_csrs(config, **x),
    "list_extensions": lambda config, **x: list_extensions(config),
    "get_extension_details": lambda config, **x: get_extension_details(config, **x),
    "get_instruction_details": lambda config, **x: get_instruction_details(config, **x),
    "get_csr_details": lambda config, **x: get_csr_details(config, **x),
    "get_stats": lambda config, **x: get_stats(config),
    "search_isa_manual": lambda config, **x: search_isa_manual(**x)
}

SYSTEM_BASE = """You are a helpful RISC-V ISA assistant with access to a comprehensive database of instructions, CSRs, and extensions from the RISC-V Unified Database (UDB).

Use the available tools to look up accurate information. Provide clear, technical explanations with relevant details like encodings, assembly syntax, and extension dependencies."""

SYSTEM_LOCAL_ONLY = SYSTEM_BASE + """

You only have access to local database tools. Use them to answer questions about RISC-V instructions, CSRs, and extensions."""

SYSTEM_WITH_EXTERNAL = SYSTEM_BASE + """

DATA SOURCES:
- Local Database (PRIMARY): search_instructions, search_csrs, get_instruction_details, get_csr_details, get_extension_details, list_extensions, get_stats
- External (SECONDARY): search_isa_manual - searches the official RISC-V ISA Manual on GitHub

IMPORTANT - SEARCH ORDER:
1. ALWAYS search the local database first using the appropriate local tools
2. Review what the local database returned (instructions, CSRs, extensions found)
3. ONLY THEN, if needed, use search_isa_manual to find additional context
4. When using external results, clearly state: "According to the official RISC-V ISA Manual (external source): ..."

Use search_isa_manual ONLY when:
- Local tools returned no results or incomplete information for the query
- User explicitly asks about design rationale, history, or "why" questions
- User asks about topics not covered by instruction/CSR/extension data (e.g., memory model, ABI, toolchain)

NEVER skip the local database search. Always check local data first."""


def ask(question, config, enable_external, request: gr.Request):
    if not API_KEY:
        return "API key not configured."
    if not _config_data:
        return "Data not loaded."

    # Rate limiting by IP
    user_id = request.client.host if request else "unknown"
    user = user_usage[user_id]

    if user["count"] >= MAX_QUESTIONS_PER_USER:
        return f"You've reached the limit of {MAX_QUESTIONS_PER_USER} questions. Thank you for trying the RISC-V ISA Chatbot!"

    user["count"] += 1
    remaining = MAX_QUESTIONS_PER_USER - user["count"]

    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        messages = [{"role": "user", "content": f"[Using {config} profile] {question}"}]

        # Track state for enforced ordering
        local_tool_used = False
        local_results_summary = []

        # Start with local tools only
        if enable_external:
            system_prompt = SYSTEM_WITH_EXTERNAL
            # Initially only provide local tools; external added after local is used
            current_tools = LOCAL_TOOLS.copy()
        else:
            system_prompt = SYSTEM_LOCAL_ONLY
            current_tools = LOCAL_TOOLS.copy()

        response = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=4096, system=system_prompt, tools=current_tools, messages=messages)

        while response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name

                    # Check if this is a local tool
                    if tool_name in LOCAL_TOOL_NAMES:
                        local_tool_used = True
                        fn = TOOL_FNS.get(tool_name)
                        if fn:
                            result = fn(config, **block.input)
                            # Summarize local results for context
                            local_results_summary.append({"tool": tool_name, "input": block.input, "result_preview": str(result)[:200]})
                        else:
                            result = {"error": "unknown tool"}

                    elif tool_name == "search_isa_manual":
                        # External tool - only allow if local was searched first
                        if not local_tool_used:
                            result = {
                                "error": "You must search the local database first before using external search.",
                                "hint": "Use search_instructions, search_csrs, or other local tools first."
                            }
                        else:
                            fn = TOOL_FNS.get(tool_name)
                            # Add local context to the result
                            result = fn(config, **block.input)
                            result["local_context"] = f"Local database was searched first. Found: {len(local_results_summary)} local results."
                    else:
                        result = {"error": "unknown tool"}

                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result, default=str)})

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # After local tool is used, add external tool if enabled
            if enable_external and local_tool_used and EXTERNAL_TOOL not in current_tools:
                current_tools = LOCAL_TOOLS + [EXTERNAL_TOOL]

            response = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=4096, system=system_prompt, tools=current_tools, messages=messages)

        answer = "".join(b.text for b in response.content if hasattr(b, "text"))

        # Add source indicator
        source_note = "Local DB only" if not enable_external else "Local DB + ISA Manual"
        return f"{answer}\n\n---\n*Profile: {config} | Source: {source_note} | Questions remaining: {remaining}/{MAX_QUESTIONS_PER_USER}*"

    except anthropic.AuthenticationError:
        user["count"] -= 1
        return "API key error."
    except Exception as e:
        user["count"] -= 1
        return f"Error: {e}"


print("Loading RISC-V data...")
build_all_indices()

# Get stats for display
total_inst = sum(len(d.get("inst_index", [])) for d in _config_data.values())
total_csr = sum(len(d.get("csr_index", [])) for d in _config_data.values())
total_ext = max((len(d.get("ext_index", [])) for d in _config_data.values()), default=0)

print(f"Loaded configs: {', '.join(AVAILABLE_CONFIGS)}")

custom_css = """
.answer-box {
    min-height: 300px;
    padding: 16px;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    overflow-y: auto;
}
"""

with gr.Blocks(title="RISC-V ISA Chatbot", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(f"""
# RISC-V ISA Chatbot

Ask questions about RISC-V instructions, CSRs, and extensions. Powered by Claude Haiku 4.5.

**Data Source:** [RISC-V Unified Database (UDB)](https://github.com/riscv-software-src/riscv-unified-db) - the centralized machine-readable RISC-V ISA specification.

**Profiles:** {', '.join(AVAILABLE_CONFIGS)} | **Limit:** {MAX_QUESTIONS_PER_USER} questions per user
""")

    with gr.Row():
        config = gr.Radio(
            choices=AVAILABLE_CONFIGS if AVAILABLE_CONFIGS else ["RV64"],
            value=DEFAULT_CONFIG if DEFAULT_CONFIG else "RV64",
            label="ISA Profile",
            info="Select RISC-V base architecture"
        )
        enable_external = gr.Checkbox(
            label="Search ISA Manual",
            value=False,
            info="Enable external search of official RISC-V ISA Manual (GitHub) when local data is insufficient"
        )

    question = gr.Textbox(label="Question", placeholder="Ask about RISC-V instructions, CSRs, or extensions...", lines=2)
    submit = gr.Button("Ask", variant="primary")
    gr.Markdown("**Answer**")
    answer = gr.Markdown(value="*Ask a question above...*", elem_classes=["answer-box"])

    gr.Examples(
        examples=[
            ["What instructions are in the M extension?"],
            ["Explain the mstatus CSR and its fields"],
            ["How does the ADD instruction work?"],
            ["List all vector extensions"],
        ],
        inputs=question,
    )

    submit.click(fn=ask, inputs=[question, config, enable_external], outputs=answer)
    question.submit(fn=ask, inputs=[question, config, enable_external], outputs=answer)

    gr.Markdown("""
---
*Data from [RISC-V Unified Database](https://github.com/riscv-software-src/riscv-unified-db) by RISC-V International*
""")

if __name__ == "__main__":
    demo.launch()
