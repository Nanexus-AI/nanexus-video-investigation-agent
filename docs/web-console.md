# Nanexus Local Web Investigation Console (V0.1)

Operator guide for the single-page Investigation Console.

## Dependencies

On the investigation host:

```bash
cd nanexus-video-investigation-agent
uv sync --locked --no-dev
```

Also required for live investigations:

- **Python 3.12**
- **Local Ollama** with planning model `qwen3:14b` (and optionally vision model `moondream:1.8b`)
- **Authenticated Frigate** access on port **8971** (V0.1 verified against Frigate **0.17.1** as a historical test baseline; for new installs prefer Frigate’s security-maintained releases and verify Nanexus against your chosen version)
- A private Frigate env file (typically `.local/frigate.env`) — start from [`.env.example`](../.env.example). Use a **restricted** Frigate account (read-only / viewer-class credentials). Do not use admin credentials in Nanexus configuration.

### Configuration loading (important)

| Settings | How they are applied |
|----------|----------------------|
| `NANEXUS_FRIGATE_*` | Private file whose path is given by **`NANEXUS_FRIGATE_ENV`** (default `.local/frigate.env`) |
| `NANEXUS_WEB_*`, `NANEXUS_OLLAMA_*`, `NANEXUS_CHAT_MODEL`, `NANEXUS_VISUAL_MODEL`, `NANEXUS_TZ`, `NANEXUS_ENABLE_OBSERVE`, and optional override of `NANEXUS_FRIGATE_ENV` itself | **Process environment** only (`WebConfig.from_env`) |

```bash
mkdir -p .local
cp .env.example .local/frigate.env
# edit Frigate keys in .local/frigate.env (secrets stay private)
# optional: export NANEXUS_FRIGATE_ENV=/absolute/or/relative/path/to/frigate.env

# Optional Web/Ollama overrides — must be exported (example):
# export NANEXUS_WEB_PORT=8787
# export NANEXUS_CHAT_MODEL=qwen3:14b
```

Copying `.env.example` to `.local/frigate.env` does **not** by itself activate Web/Ollama settings. Do not commit passwords, tokens, or private certificate files.

## Start (localhost — supported)

```bash
uv run --locked --no-dev nanexus-web
# or: uv run --locked --no-dev python -m nanexus_investigation.web.app
```

Default bind: `http://127.0.0.1:8787`

`nanexus-web` starts the HTTP server directly (no `--help` subcommand). Check the health strip / `GET /api/health`:

- `ok: true` — Web process is running (liveness)
- `frigate.ready` — Frigate configuration and reachability readiness
- `ollama_chat` / `ollama_visual` — local model readiness
- `cameras` — configured allowlist (empty when unset; fail-closed)

Missing Frigate configuration must show `frigate.ready: false` with a readable detail, not an HTTP 500.

## Investigation workflow

1. Enter a natural-language question.
2. Select one or more **authorized** cameras.
3. Set start/end times (browser local clock; converted to UTC for the Case).
4. Click **Investigate** (one investigation at a time).
5. Wait for status `completed`, `partial`, or `failed`.
6. Read Summary, Key Findings, Timeline, Selected Visual Evidence, and Important Limitations.
7. Expand **Evidence details & provenance** for claim IDs, receipts, and hashes.
8. Use **Recent Cases** to reopen a saved Case (at most 10 listed; older Cases remain in storage).

### Reopening a Case

- Restores question, cameras, and time window into the request form.
- Shows the saved results and any retained snapshot.
- Does **not** re-run the Agent, Frigate, Ollama, or VLM.
- Editing the form and clicking **Investigate** starts a **new** Case; the historical Case is unchanged.

### Case statuses

| Status | Meaning |
|--------|---------|
| `completed` | Investigation finished with an accepted report for the executed scope |
| `partial` | Accepted report with incomplete coverage, early stop, or other disclosed limitations (evidence already obtained remains available) |
| `failed` | No accepted report usable for the operator outcome (for example interrupted before finalization) |

A Case may be `partial` after conservative anti-repeat blocking even when detections, snapshots, or VLM text were already registered. That is expected V0.1 behavior, not silent deletion of evidence.

### Selective snapshots

Snapshot download and local VLM interpretation run only when the investigation selects media tools. Ordinary event/temporal searches often show: “No snapshots were retrieved for this investigation.” That does not mean Frigate has no snapshots—only that this Case did not retain any.

## Security boundary

**Supported:** localhost-only single-operator use (`NANEXUS_WEB_HOST` default `127.0.0.1`).

The browser talks only to Nanexus. Frigate and Ollama credentials stay on the server. Retained media is served only when bound to the Case’s evidence and integrity-checked.

### Temporary LAN review (not production auth)

For short UI review from another machine on a private LAN, you may bind to a host address and set an explicit client allowlist. Replace the example addresses with your own:

```bash
export NANEXUS_WEB_HOST=10.0.0.10
export NANEXUS_WEB_PORT=8787
export NANEXUS_WEB_ALLOWED_CLIENTS=10.0.0.20,10.0.0.10,127.0.0.1
uv run --locked --no-dev nanexus-web
```

This IP allowlist is a temporary restriction, **not** multi-user authentication. Prefer SSH local forwarding to `127.0.0.1:8787` when possible. Do not bind to `0.0.0.0` for V0.1.

```powershell
ssh -L 8787:127.0.0.1:8787 USER@HOST
# then open http://127.0.0.1:8787/
```

## Notes

- Process restart during a run can leave a Case interrupted — start a new investigation.
- Model text is escaped in the UI; Claims are durable artifacts.
- VLM interpretations are qualified model opinion, not independently verified physical-world facts.
- See [v0.1-as-built.md](v0.1-as-built.md) and the project README for product scope and limitations.
