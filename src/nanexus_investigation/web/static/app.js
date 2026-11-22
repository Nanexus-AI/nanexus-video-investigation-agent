(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    pollTimer: null,
    caseId: null,
    timezone: "America/Toronto",
    lastResult: null,
    viewingSaved: false,
  };

  function showError(msg) {
    const el = $("error");
    if (!msg) {
      el.classList.remove("show");
      el.textContent = "";
      return;
    }
    el.textContent = msg;
    el.classList.add("show");
  }

  function esc(text) {
    return String(text ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function toLocalInputValue(date) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  function utcIsoToLocalInput(iso) {
    // Date parses the instant; getHours/getMinutes use the browser's zone for that date (DST-aware).
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return toLocalInputValue(d);
  }

  function localInputToIso(value) {
    // datetime-local is wall time in the browser's local zone.
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) throw new Error("invalid date/time");
    return d.toISOString();
  }

  function setViewingSaved(caseId) {
    state.viewingSaved = !!caseId;
    const hint = $("viewing-hint");
    const resetBtn = $("new-investigation");
    if (!caseId) {
      hint.hidden = true;
      hint.textContent = "";
      resetBtn.hidden = true;
      return;
    }
    hint.hidden = false;
    hint.textContent = `Viewing saved investigation · ${caseId}. Edit the form and click Investigate to start a new Case; the saved Case is not changed.`;
    resetBtn.hidden = false;
  }

  function applyRequestForm(payload) {
    const result = payload.result || payload;
    const question = result.question || payload.question || "";
    const scope = result.scope || {};
    $("question").value = question;
    const cams = Array.isArray(scope.cameras) ? scope.cameras : [];
    const sel = $("cameras");
    Array.from(sel.options).forEach((opt) => {
      opt.selected = cams.length ? cams.includes(opt.value) : opt.selected;
    });
    if (scope.start_utc) $("start").value = utcIsoToLocalInput(scope.start_utc);
    if (scope.end_utc) $("end").value = utcIsoToLocalInput(scope.end_utc);
  }

  function resetToNewInvestigationDefaults() {
    stopPoll();
    state.caseId = null;
    state.lastResult = null;
    setViewingSaved(null);
    setStatus("idle");
    $("summary").textContent = "Submit a question to begin.";
    renderList($("findings"), [], "None yet.");
    $("findings-note").textContent = "";
    renderList($("timeline"), [], "None yet.");
    $("timeline-note").textContent = "";
    $("media").innerHTML = `<div class="muted">None yet.</div>`;
    renderList($("limitations"), [], "None yet.");
    $("relations-block").innerHTML = "";
    $("technical").textContent = "";
    const now = new Date();
    const hourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    $("end").value = toLocalInputValue(now);
    $("start").value = toLocalInputValue(hourAgo);
    // Keep question/cameras; operator may reuse them for a fresh run.
    $("busy-hint").textContent = "Form reset for a new investigation.";
  }

  async function api(path, options = {}) {
    const resp = await fetch(path, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });
    let data = null;
    const text = await resp.text();
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = { detail: text }; }
    if (!resp.ok) {
      const detail = data && data.detail ? data.detail : resp.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function setStatus(status) {
    const el = $("status-area");
    const cls = ["running", "completed", "partial", "failed", "unsupported"].includes(status)
      ? status
      : "";
    el.innerHTML = `<span class="status-chip ${cls}">${esc(status || "idle")}</span>`;
  }

  function renderList(el, items, empty) {
    if (!items || !items.length) {
      el.innerHTML = `<li class="muted">${esc(empty)}</li>`;
      return;
    }
    el.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
  }

  function renderResult(payload, opts = {}) {
    const result = payload.result || payload;
    const expandTimeline = !!opts.expandTimeline;
    const expandFindings = !!opts.expandFindings;
    state.lastResult = payload;
    setStatus(result.status || payload.status || "unknown");
    $("summary").textContent = result.summary || "";

    const findingRows = expandFindings
      ? (result.findings_full || result.findings || [])
      : (result.findings || []);
    const findings = findingRows.map((f) => {
      const lim = (f.limitations || []).length
        ? `<div class="hint">${esc((f.limitations || []).join(" · "))}</div>`
        : "";
      return `<strong>${esc(f.headline)}</strong><div>${esc(f.detail)}</div>${lim}`;
    });
    renderList($("findings"), findings, "No key findings.");
    const fTotal = result.findings_total || findingRows.length;
    const fShown = findingRows.length;
    let findingsNote = "";
    if (fTotal > fShown) {
      findingsNote = `Showing ${fShown} of ${fTotal} findings.`;
    } else if (fTotal) {
      findingsNote = `${fTotal} finding(s).`;
    }
    if (fTotal > (result.findings || []).length && !expandFindings) {
      findingsNote += " ";
      $("findings-note").innerHTML =
        `${esc(findingsNote)}<button type="button" class="linkish" id="expand-findings">Show all findings</button>`;
      $("expand-findings").addEventListener("click", () =>
        renderResult(state.lastResult, { expandFindings: true, expandTimeline })
      );
    } else {
      $("findings-note").textContent = findingsNote;
    }

    const timelineRows = expandTimeline
      ? (result.timeline_full || result.timeline || [])
      : (result.timeline || []);
    const timeline = timelineRows.map((t) =>
      `<strong>${esc(t.t_display || t.t)}</strong> · ${esc(t.label)} · ${esc(t.camera)}`
    );
    renderList($("timeline"), timeline, "No timeline events.");
    const total = result.timeline_total || 0;
    const shown = timelineRows.length;
    let timelineNote =
      total > shown
        ? `Showing ${shown} of ${total} registered detections (not exhaustive of all possible world events).`
        : total
          ? `${total} registered detection(s) in the authorized query result.`
          : "";
    if (total > (result.timeline || []).length && !expandTimeline) {
      $("timeline-note").innerHTML =
        `${esc(timelineNote)} <button type="button" class="linkish" id="expand-timeline">Show all detections</button>`;
      $("expand-timeline").addEventListener("click", () =>
        renderResult(state.lastResult, { expandTimeline: true, expandFindings })
      );
    } else {
      $("timeline-note").textContent = timelineNote;
    }

    const mediaEl = $("media");
    const media = result.media || [];
    if (!media.length) {
      const emptyMsg =
        result.media_empty_message ||
        "No snapshots were retrieved for this investigation.";
      mediaEl.innerHTML = `<div class="muted">${esc(emptyMsg)}</div>`;
    } else {
      mediaEl.innerHTML = media
        .map((m) => {
          const img = m.retained_media_id
            ? `<img alt="Retained snapshot" src="/api/investigations/${encodeURIComponent(result.case_id)}/media/${encodeURIComponent(m.retained_media_id)}" />`
            : "";
          const lim = (m.limitations || []).map(esc).join(" · ");
          return `<div class="media-card">${img}<div class="caption">${esc(m.visual_text || "Snapshot retained")}</div><div class="hint">${lim}</div></div>`;
        })
        .join("");
    }

    const limitations = [
      ...(result.limitations || []),
      ...(result.unresolved_questions || []).map((q) => `Unresolved: ${q}`),
    ].map(esc);
    renderList($("limitations"), limitations, "No material limitations recorded.");

    const rel = (result.relations || [])
      .map((r) => {
        const line =
          r.detail ||
          `${r.left_label} near ${r.right_label} — ${r.gap_seconds}s apart (within ${r.window_seconds}s)`;
        return `<li>${esc(line)}</li>`;
      })
      .join("");
    const qual = result.relation_qualification
      ? `<div class="hint">${esc(result.relation_qualification)}</div>`
      : "";
    $("relations-block").innerHTML = rel
      ? `<h3>Nearby activity</h3>${qual}<ul class="relations">${rel}</ul>`
      : "";

    const tech = {
      ...(result.technical || {}),
      media_refs: (result.media || []).map((m) => ({
        evidence_id: m.evidence_id,
        retained_media_id: m.retained_media_id,
        sha256_prefix: m.sha256_prefix,
        claim_id: m.claim_id,
      })),
      timeline_ids: (result.timeline_full || result.timeline || []).map((t) => ({
        t: t.t,
        label: t.label,
        camera: t.camera,
        observation_id: t.observation_id,
        native_record_id: t.native_record_id,
      })),
      finding_refs: (result.findings_full || result.findings || []).map((f) => f.refs),
    };
    $("technical").textContent = JSON.stringify(tech, null, 2);
  }

  async function refreshCases() {
    const data = await api("/api/investigations?limit=10");
    const el = $("cases");
    if (!data.cases || !data.cases.length) {
      el.innerHTML = `<li class="muted">No cases yet.</li>`;
      return;
    }
    el.innerHTML = data.cases
      .map(
        (c) =>
          `<li><button type="button" class="linkish" data-case="${esc(c.case_id)}">${esc(c.case_id)}</button> · <span class="status-chip ${esc(c.status)}">${esc(c.status)}</span> · ${esc(c.question)}</li>`
      )
      .join("");
    el.querySelectorAll("button[data-case]").forEach((btn) => {
      btn.addEventListener("click", () => openCase(btn.getAttribute("data-case")));
    });
  }

  async function openCase(caseId) {
    showError("");
    stopPoll();
    $("investigate").disabled = false;
    $("busy-hint").textContent = "";
    const data = await api(`/api/investigations/${encodeURIComponent(caseId)}`);
    state.caseId = caseId;
    applyRequestForm(data);
    setViewingSaved(caseId);
    if (data.result) renderResult(data);
    else {
      setStatus(data.status);
      $("summary").textContent = data.error || "Case has no viewable result yet.";
    }
  }

  function stopPoll() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  async function pollCase(caseId) {
    const data = await api(`/api/investigations/${encodeURIComponent(caseId)}`);
    setStatus(data.status);
    if (data.status === "running") {
      $("summary").textContent = "Investigation running…";
      return;
    }
    stopPoll();
    $("investigate").disabled = false;
    $("busy-hint").textContent = "";
    if (data.result) {
      applyRequestForm(data);
      setViewingSaved(caseId);
      renderResult(data);
    } else {
      $("summary").textContent = data.error || "Finished without a result view.";
    }
    await refreshCases();
  }

  async function investigate() {
    showError("");
    const question = $("question").value.trim();
    const selected = Array.from($("cameras").selectedOptions).map((o) => o.value);
    let startIso;
    let endIso;
    try {
      startIso = localInputToIso($("start").value);
      endIso = localInputToIso($("end").value);
    } catch (err) {
      showError("Provide valid start and end date/time values.");
      return;
    }
    if (!question || !selected.length) {
      showError("Question and at least one camera are required.");
      return;
    }
    // Always create a NEW Case from current form values; never mutate a viewed Case.
    const priorViewed = state.viewingSaved ? state.caseId : null;
    setViewingSaved(null);
    $("investigate").disabled = true;
    $("busy-hint").textContent = priorViewed
      ? `Starting new investigation (saved Case ${priorViewed} unchanged)…`
      : "Starting…";
    try {
      const created = await api("/api/investigations", {
        method: "POST",
        body: JSON.stringify({
          question,
          camera_ids: selected,
          start_utc: startIso,
          end_utc: endIso,
        }),
      });
      state.caseId = created.case_id;
      setStatus("running");
      $("summary").textContent = "Investigation running…";
      stopPoll();
      state.pollTimer = setInterval(() => {
        pollCase(created.case_id).catch((err) => showError(err.message));
      }, 2000);
      await pollCase(created.case_id);
    } catch (err) {
      $("investigate").disabled = false;
      $("busy-hint").textContent = "";
      showError(err.message || String(err));
    }
  }

  async function boot() {
    const now = new Date();
    const hourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    $("end").value = toLocalInputValue(now);
    $("start").value = toLocalInputValue(hourAgo);

    try {
      const health = await api("/api/health");
      state.timezone = health.timezone || state.timezone;
      $("tz-hint").textContent = `Operator inputs use this browser's local clock; server timezone label: ${state.timezone}. Values are converted to UTC for the investigation.`;
      $("health").textContent = `Frigate ${health.frigate?.ready ? "ready" : "not ready"} · Chat ${health.ollama_chat?.ready ? health.ollama_chat.model : "unavailable"} · Vision ${health.ollama_visual?.ready ? health.ollama_visual.model : "unavailable"} · Cameras ${((health.cameras || []).join(", ")) || "(none)"}`;
      const camData = await api("/api/cameras");
      const sel = $("cameras");
      sel.innerHTML = (camData.cameras || [])
        .map((c) => `<option value="${esc(c)}" selected>${esc(c)}</option>`)
        .join("");
      await refreshCases();
    } catch (err) {
      showError(`Failed to load console health: ${err.message}`);
    }

    $("investigate").addEventListener("click", investigate);
    $("new-investigation").addEventListener("click", () => {
      showError("");
      resetToNewInvestigationDefaults();
    });
  }

  boot();
})();
