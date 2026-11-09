# Schematic Agent

`schematic-agent` parses electrical schematic PDFs and produces structured
engineering data: a bill of materials (BOM), a netlist, detected components,
PLC logic data, issues with confidence values, and a human-readable report.

The agent is a single self-contained worker with a narrow public API. It does
not create external child agents and can be embedded in a larger orchestration
system as one indivisible executor.

## What it does

Input: a vector PDF of an electrical schematic.

Output: component detections, BOM (from tables or derived from detected
components), electrical nets, PLC inputs/outputs/coils/contacts/interlocks,
issues for low-confidence or ambiguous regions, and a `report.md` summary.

The pipeline is deterministic by default. An optional local vision-language
model (Ollama, `qwen3-vl:2b`) is queried only for point-wise ambiguous
triggers; it never replaces the deterministic result.

## Architecture

```text
PDF extraction (PyMuPDF + pdfplumber)
        │
        ▼
Geometry (segments, nodes, crossings, labels; spatial index)
        │
        ▼
Components and symbols (templates, ref designations)
        │
        ▼
BOM + netlist (pdfplumber tables, networkx graph)
        │
        ▼
PLC analysis (I/O, coils, contacts, interlocks, logic blocks)
        │
        ▼
Optional local VLM (Ollama qwen3-vl:2b, point-wise triggers only)
        │
        ▼
Issues, consistency score, report.md
```

| Module | Purpose |
|---|---|
| `schematic_agent/parser/pdf_extract.py` | text + bboxes, vector primitives, tables, raster detection |
| `schematic_agent/geometry.py` | segments, nodes, crossings, net labels, frame/table exclusion |
| `schematic_agent/spatial.py` | uniform-grid indexes for detectors and intersection search |
| `schematic_agent/components.py` | symbol templates and reference designations |
| `schematic_agent/bom.py` | BOM from pdfplumber tables, normalization, schematic reconciliation |
| `schematic_agent/netlist.py` | electrical net graph and cross-page labels |
| `schematic_agent/plc.py` | inputs/outputs/coils/contacts/interlocks/logic blocks |
| `schematic_agent/vlm.py` | point-wise calls to the local VLM (Ollama) |
| `schematic_agent/issues.py` | single issue-type vocabulary and helpers |
| `schematic_agent/report.py` | `report.md` assembly |
| `schematic_agent/pipeline.py` | `run()`, modes, artifacts, statuses |

## Installation

Requires Python >= 3.10.

```bash
git clone <repository-url> schematic_agent_public
cd schematic_agent_public
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

Dependencies: `pydantic>=2`, `pymupdf>=1.24`, `pdfplumber>=0.11`,
`networkx>=3`; tests use `pytest>=8`.

The optional VLM stage needs a local Ollama instance:

```bash
ollama pull qwen3-vl:2b
```

Recommended Ollama environment for lower memory use and faster inference:

```bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q4_0
```

The agent and Ollama run on the same machine; the default endpoint is
`http://127.0.0.1:11434`.

## Quickstart

### Python API

```python
from schematic_agent import Task, TaskOptions, TaskPayload, run

task = Task(
    task_id="task-001",
    payload=TaskPayload(
        pdf_path="/data/input/schema.pdf",
        document_id="doc_001",
        options=TaskOptions(
            mode="full",
            use_vlm=True,
            extract_bom=True,
            extract_netlist=True,
            analyze_plc_logic=True,
        ),
    ),
)

result = run(task)
print(result.status)
print(result.artifacts["report"])
```

The public API also exposes `submit(task) -> str`, `get_result(task_id)`,
`health_check()`, and `describe_capabilities()`.

| Function | Contract |
|---|---|
| `run(task) -> Result` | runs the task synchronously |
| `submit(task) -> str` | runs the task, keeps the result in memory, returns `task_id` |
| `get_result(task_id) -> Result` | returns a stored result; raises `KeyError` for unknown IDs |
| `health_check() -> dict` | `{"status": "ok"}` |
| `describe_capabilities() -> dict` | task type, supported modes, VLM parameters |

### CLI

```bash
schematic-agent parse /path/to/schema.pdf --mode full --output /tmp/out
```

| Argument | Purpose |
|---|---|
| `pdf` | path to the PDF file |
| `--mode` | `full`, `no_vlm`, `bom_only`, `netlist_only`, `issue_review` |
| `--no-vlm` | disables the VLM entirely (with `--mode full` behaves as `no_vlm`) |
| `--output` | artifact directory (overrides `SCHEMATIC_AGENT_OUTPUT_DIR`) |
| `--task-id` | task identifier (default `cli-task`) |
| `--document-id` | document identifier (default: PDF name without extension) |

The CLI prints the `Result` as JSON and exits with code `0` unless the status
is `failed`, in which case it exits with `1`.

## Modes

| Mode | BOM | Netlist | PLC | VLM |
|---|---|---|---|---|
| `full` | yes | yes | yes | yes, when `use_vlm=True` and VLM is enabled |
| `no_vlm` | yes | yes | yes | no |
| `bom_only` | yes | no | no | no |
| `netlist_only` | no | yes | no | no |
| `issue_review` | yes | yes | yes | yes |

`issue_review` currently runs the same stages as `full`. No VLM request is
made when `mode=no_vlm`, `use_vlm=False`, or `VlmSettings.enabled=False`.

## Artifacts

Artifacts are written to `<output_dir>/<document_id>/`. Eight files are
produced:

| File | Content |
|---|---|
| `bom.json` | bill of materials: `{"items": [BomItem], "source": ...}` |
| `bom.csv` | BOM as CSV |
| `netlist.json` | electrical nets |
| `components.json` | detected components |
| `issues.json` | issues and low-confidence regions |
| `plc_analysis.json` | PLC logic data |
| `report.md` | final report |
| `vlm_calls.log` | VLM call log (JSON Lines, append) |

`Result.artifacts` contains absolute paths to all eight files. Except for a
missing input file, artifacts are created even for `failed` results so callers
can distinguish "no data" from an operational failure.

## Configuration

| Environment variable | Purpose | Default |
|---|---|---|
| `SCHEMATIC_AGENT_OUTPUT_DIR` | artifact directory | `/output` |
| `SCHEMATIC_AGENT_DOC_TIMEOUT_S` | soft document-wide timeout, seconds | `600` |
| `SCHEMATIC_AGENT_STAGE_TIMEOUT_S` | per-stage guard timeout, seconds | `120` |
| `SCHEMATIC_AGENT_MAX_CANDIDATES_PER_PAGE` | detector candidate cap per page | `10000` |
| `SCHEMATIC_AGENT_MAX_SEGMENTS_PER_DETECTOR` | input segment cap per detector | `5000` |

Invalid or non-positive values are ignored and replaced by defaults.

VLM defaults (`schematic_agent/config.py`, `VlmSettings`): endpoint
`http://127.0.0.1:11434`, model `qwen3-vl:2b`, crop size 512–768 px,
`max_calls=50`, `timeout_s=120`, `enabled=True`. Geometry tolerances live in
`config.Tolerances`.

## Vision-language model (VLM)

The VLM is used only for point-wise triggers: unrecognized symbols, complex
relays and ICs (`U*`, `DA*`), ambiguous wire crossings without a junction, and
empty or unreadable BOM cells. Only image crops of 512×512 px (up to
768×768 px) are sent. Calls are strictly sequential (VRAM protection), the
response must be strict JSON, and `num_predict` is bounded to keep generation
short. A budget of `max_calls` limits the total number of requests per run.

If the VLM is unavailable or misbehaves, the deterministic result is kept and
an issue is recorded:

| Situation | Issue | Behavior |
|---|---|---|
| VLM unreachable / timeout | `vlm_unavailable` | deterministic result preserved |
| HTTP 404 (model missing) | `vlm_model_missing` | further VLM calls stop |
| Invalid or empty JSON | `vlm_bad_json` | exactly one retry, then issue |
| `max_calls` exhausted | `vlm_budget_exceeded` | remaining triggers are skipped |

Known limitation: `qwen3-vl` thinking mode can return an empty `response`
field for crossing-related prompts. This is treated as `vlm_bad_json`
(honest degradation): the deterministic geometry result is preserved, the
issue is reported, and no fabricated data is produced.

Real VLM calls are never made by the automated tests; the HTTP transport is
injected and mocked.

## Reliability

Statuses: `completed`, `completed_with_issues`, `partial`, `failed`.

| Situation | Status |
|---|---|
| missing file | `failed`, `error="file_not_found: ..."` |
| corrupted PDF | `failed`, `error="parse_error: ..."` |
| empty file | `failed`, `error="empty_file"` |
| page failure (`page_failed`) | `partial` |
| document timeout (`doc_timeout`) | `partial` |
| stage guard timeout (`stage_timeout`) | `partial` |
| detector cap reached (`detector_capped`) | `partial` |
| raster-only page (`raster_only_page`) | `completed_with_issues` |
| other issues | `completed_with_issues` |
| no issues | `completed` |

A failing page does not interrupt the remaining pages. Wire crossings without
a junction dot do not create an electrical connection; the decoupling is
recorded in `GeometryResult.crossings`.

Guards emit an issue and never abort the pipeline:

| Guard | Issue | Behavior |
|---|---|---|
| `stage_timeout_s` | `stage_timeout` | stage marked, status `partial` |
| `max_candidates_per_page` | `detector_capped` | first N candidates processed |
| `max_segments_per_detector` | `detector_capped` | first N segments processed |

## Performance

Detectors and intersection search use uniform-grid indexes from
`schematic_agent/spatial.py`; only geometrically neighboring segments are
examined instead of all pairs or triples. Pair queries drop from O(n²) to
about O(n·k), and triangle enumeration from O(n³) to about O(n·k²), where
`k` is the number of neighbors in the local grid neighborhood. Triangle
candidate sets are unchanged with respect to the previous exhaustive
implementation.

## BOM fallback

If no BOM table is recognized, `bom.py` builds a BOM from detected components
and sets `source="components_fallback"` in `bom.json`. Components are
aggregated by `(ref_designation, type)`: `quantity` is the match count,
`name`/`category` come from the component type, `page` is the page of the
first component, and `confidence` is `min(confidence) * 0.9`. Components of
type `unknown` are named `[NOT IN BOM]`. Components without a
`ref_designation` are skipped. All fallback rows have
`found_on_schematic=True`; `component_not_in_bom` and
`bom_item_not_on_schematic` are not raised for the fallback. The
`bom_table_missing` issue is preserved, and `source="table"` is used when a
table exists. The fallback runs in `full`, `no_vlm`, `issue_review`, and
`bom_only`; `netlist_only` is unaffected.

## Multimodal consistency score

After BOM reconciliation, `report.py` adds a "channel reconciliation" section
to `report.md`. It reports the number of schematic components with a
`ref_designation` (A) and the number of BOM rows (B), and computes:

```text
MULTIMODAL CONSISTENCY SCORE = (1 - |A - B| / max(A, B)) * 100%
```

If the VLM ran, the share of unknown components confirmed by successful VLM
responses is also reported. Only actual artifact counts are used; with empty
channels the score is marked as undefined.

Reference designations are matched between BOM and schematic by a normalized
key (`bom.normalize_ref`: uppercase, keep only `[A-Z0-9]`), so `"c101 "` and
`"C-101"` match. The stored `ref_designation` values are not modified, and
nested designations (`C101` inside `C101HUD`) are not matched.

## Limitations

- Raster pages are not OCR'd: `raster_only_page` is recorded with action
  `OCR required`, and page results are empty.
- Built-in symbol templates are limited: resistor, capacitor, diode, terminal,
  coil/winding, contact (`SYMBOL_TEMPLATES`). Other symbols near a reference
  designation produce `Component(type="unknown")` and
  `Issue(type="unrecognized_symbol")`, which trigger the VLM.
- Component type comes from the template; the designation prefix refines the
  type only for unrecognized symbols.
- Labels `L1`/`L2`/`L3` are treated as mains phases, not inductors: the `L`
  prefix is intentionally excluded from reference-designation parsing in
  geometry.
- Cross-page labels merge nets only on exact label-name match.

## Examples

`examples/generate_examples.py` deterministically creates three PDFs:

```text
examples/simple.pdf     1 page: supply, F1, SB1, K1, HL1, L1/N, BOM (4 rows)
examples/multipage.pdf  2 pages: cross-page label L1, components R1 and R2
examples/problem.pdf    3 pages: raster page, crossing without junction, broken table
```

## Testing

```bash
python examples/generate_examples.py --output-dir examples
PYTHONPATH=. python -m pytest
PYTHONPATH=. python -m pytest -m acceptance -v
```

The automated tests use example PDFs and mocked VLM transport only.

Integration tests that exercise an external reference document are skipped
unless the `SCHEMATIC_AGENT_TEST_PDF` environment variable points to an
existing PDF. No external document is bundled with the repository:

```bash
SCHEMATIC_AGENT_TEST_PDF=/path/to/reference.pdf PYTHONPATH=. python -m pytest
```

## License

The project code is released under the MIT License. See `LICENSE`.

> **PyMuPDF license notice.** PyMuPDF is distributed under **AGPL-3.0**.
> This imposes AGPL requirements on distribution and network use of a derived
> product. Replacing PyMuPDF with an alternative parser may be required for
> some deployment scenarios. Review the AGPL obligations before public or
> commercial use.
