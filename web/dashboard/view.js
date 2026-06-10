export function createDashboardView(ctx) {
  const { state, constants, elements, helpers, actions } = ctx;
  const { OPPONENTS, OPPONENT_LABELS, OPPONENT_COLORS, OPPONENT_DESCRIPTIONS } = constants;
  const { formatDuration, formatTime, metricValue, resultLabel, setStatusPill } = helpers;
  const { openReplay, setOpponentFilter, setBusy, setStatus } = actions;
  const {
    battleList,
    benchmarkSection,
    benchmarkTab,
    connectionState,
    headerEta,
    jobEta,
    jobProgress,
    jobProgressBar,
    jobProgressLabel,
    jobProgressPercent,
    lastRefresh,
    logOutput,
    metricRows,
    opponentTabs,
    summaryCards,
    summaryTitle,
    trainSection,
    trainTab,
  } = elements;
  function clearBenchmarkStats() {
    state.summary = null;
    state.liveProgress = null;
    state.lastSummarySignature = "";
    state.lastProgressSignature = "";
    summaryCards.replaceChildren();
    metricRows.replaceChildren();
    renderEmptySummary();
    renderEmptyMetrics();
  }

  function setBenchmarkResultsVisible(visible, showMetrics = false) {
    state.benchmarkResultsVisible = Boolean(visible);
    state.metricsVisible = Boolean(showMetrics);
    if (!state.benchmarkResultsVisible) clearBenchmarkStats();
  }

  function renderEmptySummary() {
    summaryTitle.textContent = "Awaiting run";
    summaryCards.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = `
    <div class="ball"></div>
    <div class="es-title">No benchmark has run yet</div>
    <div class="es-sub">Configure opponents on the left and press <b>Benchmark</b>. Per-opponent win rates, metrics and battles will populate here.</div>
  `;
    summaryCards.append(empty);
  }

  function renderEmptyMetrics() {
    metricRows.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = `
    <div class="es-sub">Win rate | Generalization | Risk | Threat coverage | Efficiency<br>Metrics will compute once a benchmark completes.</div>
  `;
    metricRows.append(empty);
  }

  function setConnectionState(text, stateName = "idle") {
    connectionState.replaceChildren();
    const dot = document.createElement("span");
    dot.className = "dot";
    connectionState.append(dot, document.createTextNode(text));
    connectionState.dataset.state = stateName;
  }

  function setControlsTab(tabName) {
    const trainActive = tabName === "train";
    benchmarkSection.hidden = trainActive;
    trainSection.hidden = !trainActive;
    benchmarkTab.classList.toggle("active", !trainActive);
    trainTab.classList.toggle("active", trainActive);
    benchmarkTab.setAttribute("aria-selected", String(!trainActive));
    trainTab.setAttribute("aria-selected", String(trainActive));
  }

  function updateHeaderTiming(progress) {
    const eta = progress?.eta_seconds;
    setStatusPill(headerEta, "ETA", eta === null || eta === undefined ? "--" : formatDuration(eta));
    setStatusPill(lastRefresh, "SYNC", formatTime());
  }

  function renderJob(job) {
    if (!job) {
      setConnectionState("Idle", "idle");
      jobProgress.hidden = true;
      return;
    }
    renderProgress(job);
    renderLog(job);
    const label = job.id === "train" ? "Train" : "Benchmark";
    if (job.status === "running") {
      setStatus(job.paused ? `${label} paused` : `${label} running`);
      setConnectionState(job.paused ? "Paused" : "Running", job.paused ? "paused" : "running");
      setBusy(true);
    } else if (job.status === "ok") {
      setStatus(`${label} finished`);
      setConnectionState("Complete", "ok");
      setBusy(false);
    } else if (job.status === "stopped") {
      setStatus(`${label} stopped`);
      setConnectionState("Stopped", "idle");
      setBusy(false);
    } else if (job.status === "failed") {
      setStatus(`${label} failed`, true);
      setConnectionState("Failed", "failed");
      setBusy(false);
    }
    if (job.id === "benchmark") renderLiveBenchmark(job);
  }

  function renderLog(job) {
    const log = Array.isArray(job.log) && job.log.length ? job.log : ["Job started..."];
    const signature = `${job.id}:${job.log_length || log.length}:${job.last_log_line || log.at(-1) || ""}`;
    if (signature === state.lastLogSignature) return;
    state.lastLogSignature = signature;
    const isPinnedToBottom =
      logOutput.scrollHeight - logOutput.scrollTop - logOutput.clientHeight < 24;
    logOutput.textContent = log.join("\n");
    if (isPinnedToBottom || job.status === "running") {
      logOutput.scrollTop = logOutput.scrollHeight;
    }
  }

  function renderProgress(job) {
    const progress = job?.progress;
    if (
      !progress ||
      (!state.activeJob &&
        job.status !== "running" &&
        job.status !== "ok" &&
        job.status !== "failed")
    ) {
      jobProgress.hidden = true;
      updateHeaderTiming(null);
      return;
    }
    const percentDone = Math.max(0, Math.min(100, Number(progress.percent || 0)));
    const current = Number(progress.current || 0);
    const total = Number(progress.total || 0);
    const eta = progress.eta_seconds;
    jobProgress.hidden = false;
    jobProgressBar.value = percentDone;
    jobProgressPercent.textContent = `${percentDone.toFixed(1)}%`;
    const labelPrefix = job.paused ? "Paused - " : "";
    jobProgressLabel.textContent = `${labelPrefix}${progress.label || "Working"}${total ? ` (${current}/${total})` : ""}`;
    jobEta.textContent = job.paused
      ? `Elapsed ${formatDuration(progress.elapsed_seconds || 0)} | waiting to resume`
      : `Elapsed ${formatDuration(progress.elapsed_seconds || 0)} | ETA ${formatDuration(eta)}`;
    updateHeaderTiming(progress);
  }

  function liveStatsFor(opponent) {
    const statuses = state.liveProgress?.opponent_statuses || {};
    return statuses[opponent] || null;
  }

  function statsFor(opponent) {
    if (state.summary) {
      if (opponent === "all") return state.summary.overall;
      return state.summary.opponents ? state.summary.opponents[opponent] : null;
    }
    if (state.liveProgress && opponent !== "all") return liveStatsFor(opponent);
    return null;
  }

  function visibleOpponentEntries() {
    if (state.liveProgress) {
      const liveOpponents = state.liveProgress.opponents || [];
      return liveOpponents.map((opponent) => [opponent, OPPONENT_LABELS[opponent] || opponent]);
    }
    if (state.summary?.opponents) {
      const entries = OPPONENTS.filter(([opponent]) => {
        if (opponent === "all") return false;
        return Number(state.summary.opponents?.[opponent]?.total || 0) > 0;
      });
      if (entries.length) return entries;
    }
    return OPPONENTS.filter(([opponent]) => opponent !== "all");
  }

  function renderLiveBenchmark(job) {
    const progress = job.progress || {};
    state.liveProgress = progress;
    setBenchmarkResultsVisible(true, false);
    summaryTitle.textContent =
      job.status === "running"
        ? job.paused
          ? "Benchmark Paused"
          : "Benchmark Progress"
        : "Benchmark Result";
    const signature = JSON.stringify({
      status: job.status,
      current: progress.current,
      total: progress.total,
      active: progress.active_opponent,
      completed: progress.completed_opponents,
      statuses: progress.opponent_statuses,
    });
    if (signature === state.lastProgressSignature) return;
    state.lastProgressSignature = signature;
    summaryCards.replaceChildren();
    const statuses = progress.opponent_statuses || {};
    for (const [opponent, label] of visibleOpponentEntries()) {
      const stats = statuses[opponent] || {
        status: "pending",
        current: 0,
        total: progress.phase_total || 0,
        wins: 0,
        losses: 0,
        winrate_finished: 0,
        description: OPPONENT_DESCRIPTIONS[opponent],
      };
      summaryCards.append(createSummaryCard(opponent, label, stats, "live"));
    }
    renderTabs();
  }

  function createSummaryCard(opponent, label, stats, mode) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "tcard";
    card.style.setProperty("--type", OPPONENT_COLORS[opponent] || "var(--accent)");
    if (state.opponentFilter === opponent) card.classList.add("active");
    if (stats.status) card.classList.add(stats.status);
    card.addEventListener("click", () => setOpponentFilter(opponent));

    const top = document.createElement("div");
    top.className = "tcard-top";
    const title = document.createElement("span");
    title.className = "tcard-title";
    title.textContent = label;
    const statePill = document.createElement("span");
    statePill.className = "tcard-state";
    statePill.textContent = stats.status
      ? stats.status.toUpperCase()
      : mode === "final"
        ? "FINAL"
        : "LIVE";
    top.append(title, statePill);

    const rate = document.createElement("div");
    rate.className = "tcard-rate";
    const rateNum = document.createElement("span");
    rateNum.className = "num tnum";
    rateNum.textContent = Number(stats.winrate_finished || 0).toFixed(1);
    const rateUnit = document.createElement("span");
    rateUnit.className = "unit";
    rateUnit.textContent = "%";
    const rateDelta = document.createElement("span");
    rateDelta.className = "delta";
    rateDelta.textContent = "wr";
    rate.append(rateNum, rateUnit, rateDelta);

    const meta = document.createElement("div");
    meta.className = "tcard-counts";
    const current = Number(stats.current ?? stats.finished ?? stats.total ?? 0);
    const total = Number(stats.total || current || 0);
    const wins = Number(stats.wins || 0);
    const losses = Number(stats.losses || 0);
    meta.innerHTML = `<span><b>${current}</b><span class="sep"> / </span>${total}</span><span><b style="color: var(--c-win)">${wins}</b> wins</span><span><b style="color: var(--c-loss)">${losses}</b> losses</span>`;

    const progressBar = document.createElement("div");
    progressBar.className = "hpbar";
    const progressFill = document.createElement("span");
    const progressTotal = Number(stats.total || 0);
    const progressCurrent = Number(stats.current ?? stats.finished ?? progressTotal);
    progressFill.style.width = `${progressTotal ? Math.max(2, Math.min(100, (100 * progressCurrent) / progressTotal)) : 0}%`;
    progressBar.append(progressFill);

    const description = document.createElement("div");
    description.className = "tcard-desc";
    description.textContent =
      stats.description || OPPONENT_DESCRIPTIONS[opponent] || "Custom benchmark opponent.";

    card.append(top, rate, meta, description, progressBar);
    return card;
  }

  function createBenchmarkContextCard(summary) {
    const overall = summary?.overall || {};
    const finished = Number(overall.finished || overall.total || 0);
    const wins = Number(overall.wins || 0);
    const winRate = Number(overall.winrate_finished || 0);
    const hasResult = finished > 0;
    const card = document.createElement("div");
    card.className = "benchmark-context";

    const title = document.createElement("div");
    title.className = "benchmark-context-title";
    title.textContent = hasResult
      ? `Validated result: ${wins}/${finished} wins (${winRate.toFixed(1)}%)`
      : "Validated profile: CTDE MLP generalization benchmark";

    const details = document.createElement("div");
    details.className = "benchmark-context-detail";
    details.textContent =
      "Recommended default: CTDE MLP reranker, inter-agent communication enabled, Simple/Abyssal opponents, and curated ally teams. Disable curated teams to run the random-ally generalization setting.";

    card.append(title, details);
    return card;
  }

  function renderSummary(summary, showMetrics = true) {
    state.summary = summary || { opponents: {}, overall: {} };
    state.liveProgress = null;
    setBenchmarkResultsVisible(true, showMetrics);
    summaryTitle.textContent = "Final Benchmark Result";
    const signature = JSON.stringify({
      overall: state.summary.overall,
      opponents: state.summary.opponents,
      showMetrics,
    });
    if (signature !== state.lastSummarySignature) {
      state.lastSummarySignature = signature;
      summaryCards.replaceChildren();
      summaryCards.append(createBenchmarkContextCard(state.summary));
      for (const [opponent, label] of visibleOpponentEntries()) {
        const stats = statsFor(opponent) || {};
        summaryCards.append(createSummaryCard(opponent, label, stats, "final"));
      }
    }
    if (showMetrics) renderBenchmarkMetrics(state.summary.metrics || {});
    renderTabs();
  }

  const totalMetricOrder = [
    "win_rate",
    "generalization",
    "risky_no_protect",
    "primary_threat_coverage",
    "conflict",
    "consistency",
    "efficiency",
  ];
  const outcomeMetricOrder = [
    "risky_no_protect",
    "primary_threat_coverage",
    "conflict",
    "consistency",
    "average_turns",
    "illegal_rejected_move_rate",
    "ctde_top1_alignment",
    "model_inference_latency",
  ];

  const METRIC_LATEX_FORMULAS = {
    win_rate: String.raw`\(\mathrm{WinRate}=100\cdot\frac{W}{B_f}\)`,
    generalization: String.raw`\(\mathrm{Generalization}=100\cdot\frac{W_{\mathrm{nonfixed}}}{B_{\mathrm{nonfixed},f}}\)`,
    risky_no_protect: String.raw`\(\mathrm{RiskyNoProtect}=100\cdot\frac{D_{\mathrm{risk,noProtect}}}{D_{\mathrm{risk}}}\)`,
    primary_threat_coverage: String.raw`\(\mathrm{ThreatCoverage}=100\cdot\frac{D_{\mathrm{risk,covered}}}{D_{\mathrm{risk}}}\)`,
    conflict: String.raw`\(\mathrm{Conflict}=100\cdot\frac{T_{\mathrm{conflict}}}{T_{\mathrm{analysed}}}\)`,
    consistency: String.raw`\(\mathrm{Consistency}=100\cdot\frac{S_{\mathrm{consistent}}}{S_{\mathrm{planned}}}\)`,
    efficiency: String.raw`\(\mathrm{Efficiency}=\frac{\mathrm{WinRate}}{\bar{c}},\quad \bar{c}\in\{\mathrm{seconds},\mathrm{turns}\}\)`,
    average_turns: String.raw`\(\mathrm{AvgTurns}=\frac{1}{|B_{\log}|}\sum_{b\in B_{\log}}\max_t(t_b)\)`,
    average_decision_latency: String.raw`\(\mathrm{DecisionLatency}=\frac{1}{B_f}\sum_{b\in B_f}\mathrm{elapsedSeconds}_b\)`,
    illegal_rejected_move_rate: String.raw`\(\mathrm{RejectedRate}=100\cdot\frac{T_{\mathrm{rejected}}}{T_{\mathrm{analysed}}}\)`,
    ctde_top1_alignment: String.raw`\(\mathrm{CTDETop1}=100\cdot\frac{T_{\mathrm{chosen=top1}}}{T_{\mathrm{scored}}}\)`,
    model_inference_latency: String.raw`\(\mathrm{ModelLatency}=\frac{1}{T_{\mathrm{scored}}}\sum_{t\in T_{\mathrm{scored}}}\mathrm{latencyMs}_t\)`,
  };

  function metricLatexFormula(key, metric) {
    if (METRIC_LATEX_FORMULAS[key]) return METRIC_LATEX_FORMULAS[key];
    if (!metric?.formula) return "";
    return `\\(\\mathrm{${key}}=${String(metric.formula).replaceAll("_", "\\_")}\\)`;
  }

  function typesetMetricMath(node) {
    if (!node || typeof window === "undefined") return;
    const runTypeset = () => {
      if (!window.MathJax?.typesetPromise) return;
      window.MathJax.typesetPromise([node]).catch(() => {});
    };
    if (window.MathJax?.startup?.promise) {
      window.MathJax.startup.promise.then(runTypeset).catch(() => {});
    } else {
      window.requestAnimationFrame(runTypeset);
    }
  }

  function metricOpponentEntries() {
    const entries = [["all", "All"], ...visibleOpponentEntries()];
    return entries.filter(([opponent], index, allEntries) => {
      if (opponent === "all") return true;
      return allEntries.findIndex(([candidate]) => candidate === opponent) === index;
    });
  }

  function metricScope() {
    const resultKey = state.metricResultFilter || "total";
    let opponentKey = state.metricOpponentFilter || "all";
    let opponentStats =
      opponentKey === "all" ? state.summary?.overall : state.summary?.opponents?.[opponentKey];
    if (!opponentStats || Number(opponentStats.total || opponentStats.finished || 0) <= 0) {
      opponentKey = "all";
      state.metricOpponentFilter = "all";
      opponentStats = state.summary?.overall;
    }
    const metricSource =
      opponentKey === "all" ? state.summary : state.summary?.opponents?.[opponentKey];
    const source =
      resultKey === "total"
        ? metricSource
        : metricSource?.result_splits?.[resultKey] || null;
    const metrics = resultKey === "total" ? source?.metrics || {} : source?.metrics || {};
    const battleCount =
      resultKey === "total"
        ? Number(opponentStats?.finished || opponentStats?.total || 0)
        : Number(source?.battle_count || 0);
    return {
      resultKey,
      opponentKey,
      metrics,
      battleCount,
      resultLabel:
        resultKey === "wins" ? "Wins" : resultKey === "losses" ? "Losses" : "Total",
      opponentLabel: opponentKey === "all" ? "All opponents" : OPPONENT_LABELS[opponentKey] || opponentKey,
    };
  }

  function createMetricTabs(scope) {
    const wrapper = document.createElement("div");
    wrapper.className = "metric-scope-tabs";

    const resultTabs = document.createElement("div");
    resultTabs.className = "metric-tab-row";
    resultTabs.setAttribute("aria-label", "Metric result scope");
    for (const [key, label] of [
      ["total", "Total"],
      ["wins", "Wins"],
      ["losses", "Losses"],
    ]) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "metric-tab";
      tab.classList.toggle("active", scope.resultKey === key);
      tab.textContent = label;
      tab.addEventListener("click", () => {
        state.metricResultFilter = key;
        renderBenchmarkMetrics();
      });
      resultTabs.append(tab);
    }

    const opponentTabsRow = document.createElement("div");
    opponentTabsRow.className = "metric-tab-row";
    opponentTabsRow.setAttribute("aria-label", "Metric opponent scope");
    for (const [opponent, label] of metricOpponentEntries()) {
      const stats = opponent === "all" ? state.summary?.overall : state.summary?.opponents?.[opponent];
      const count = Number(stats?.finished || stats?.total || 0);
      if (opponent !== "all" && count <= 0) continue;
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "metric-tab";
      tab.style.setProperty("--type", OPPONENT_COLORS[opponent] || "var(--fg-faint)");
      tab.classList.toggle("active", scope.opponentKey === opponent);
      tab.innerHTML = `<span class="metric-tab-dot"></span>${label}<span class="metric-tab-count">${count}</span>`;
      tab.addEventListener("click", () => {
        state.metricOpponentFilter = opponent;
        renderBenchmarkMetrics();
      });
      opponentTabsRow.append(tab);
    }

    const caption = document.createElement("div");
    caption.className = "metric-scope-caption";
    caption.textContent = `${scope.resultLabel} metrics for ${scope.opponentLabel} - n=${scope.battleCount}`;
    if (scope.resultKey !== "total") {
      caption.textContent +=
        " - outcome-dependent metrics are hidden in this view to avoid tautological values.";
    }

    wrapper.append(resultTabs, opponentTabsRow, caption);
    return wrapper;
  }

  function renderBenchmarkMetrics() {
    const scope = metricScope();
    const metrics = scope.metrics || {};
    const order = scope.resultKey === "total" ? totalMetricOrder : outcomeMetricOrder;
    const availableOrder = order.filter((key) => metrics[key]);
    const countNode = document.querySelector("#metricsPanel .pcount");
    if (countNode) countNode.textContent = `${availableOrder.length} metrics`;
    metricRows.replaceChildren();
    metricRows.append(createMetricTabs(scope));
    if (!metrics || !Object.keys(metrics).length || !availableOrder.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.innerHTML = `<div class="es-sub">No metrics are available for this scope.</div>`;
      metricRows.append(empty);
      return;
    }
    for (const key of availableOrder) {
      const metric = metrics[key] || {};
      const row = document.createElement("div");
      row.className = "mcard";
      if (metric.available === false) row.classList.add("muted");

      const head = document.createElement("div");
      head.className = "mcard-head";
      const name = document.createElement("span");
      name.className = "mcard-label";
      name.textContent = metric.label || key;
      const direction = document.createElement("span");
      direction.className = `mcard-arrow${metric.direction === "lower" ? " lower" : ""}`;
      direction.textContent = metric.direction === "lower" ? "lower" : "higher";
      head.append(name, direction);

      const value = document.createElement("div");
      value.className = "mcard-value";
      const renderedValue = metricValue(metric);
      const valueMatch = /^([-+]?\d+(?:\.\d+)?)(.*)$/.exec(renderedValue);
      const num = document.createElement("span");
      num.textContent = valueMatch ? valueMatch[1] : renderedValue;
      const unit = document.createElement("span");
      unit.className = "mcard-unit";
      unit.textContent = valueMatch ? valueMatch[2].trim() : "";
      value.append(num, unit);

      const description = document.createElement("div");
      description.className = "mcard-desc";
      description.textContent = metric.measures || metric.summary || "";

      const formula = document.createElement("div");
      formula.className = "mcard-formula";
      const formulaLabel = document.createElement("span");
      formulaLabel.textContent = "Formula";
      const formulaMath = document.createElement("div");
      formulaMath.className = "mcard-formula-math";
      formulaMath.textContent = metricLatexFormula(key, metric);
      formula.append(formulaLabel, formulaMath);

      const detail = document.createElement("div");
      detail.className = "mcard-detail";
      detail.textContent = [metric.detail, metric.sample].filter(Boolean).join(" - ");

      row.append(head, value, description, formula, detail);
      metricRows.append(row);
    }
    typesetMetricMath(metricRows);
  }

  function renderTabs() {
    opponentTabs.replaceChildren();
    const entries = [["all", "All"], ...visibleOpponentEntries()];
    for (const [opponent, label] of entries) {
      const stats = statsFor(opponent) || {};
      const total = stats.total || stats.current || 0;
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "tab";
      tab.style.setProperty("--type", OPPONENT_COLORS[opponent] || "var(--fg-faint)");
      if (state.opponentFilter === opponent) tab.classList.add("active");
      const dot = document.createElement("span");
      dot.className = "tab-dot";
      const count = document.createElement("span");
      count.className = "tab-count";
      count.textContent = `(${total || 0})`;
      tab.append(dot, document.createTextNode(label), count);
      tab.addEventListener("click", () => setOpponentFilter(opponent));
      opponentTabs.append(tab);
    }
  }

  function battleSignature(battles) {
    return JSON.stringify({
      selected: state.selectedBattle,
      opponent: state.opponentFilter,
      result: state.resultFilter,
      count: battles.length,
      first: battles[0]?.battle_tag || "",
      last: battles.at(-1)?.battle_tag || "",
    });
  }

  function renderBattles(battles) {
    const signature = battleSignature(battles);
    if (signature === state.lastBattleSignature) return;
    state.lastBattleSignature = signature;
    battleList.replaceChildren();
    if (!battles.length) {
      const empty = document.createElement("div");
      empty.className = "battle-empty";
      empty.textContent = "No battles match the selected filters.";
      battleList.append(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const battle of battles) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "battle-row";
      if (state.selectedBattle === battle.battle_tag) row.classList.add("active");
      row.disabled = !battle.replay_available;

      const index = document.createElement("span");
      index.className = "battle-idx";
      index.textContent = battle.battle_idx
        ? `#${String(battle.battle_idx).padStart(3, "0")}`
        : "--";

      const summary = document.createElement("span");
      summary.className = "battle-mid";
      const tag = document.createElement("span");
      tag.className = "battle-tag";
      tag.textContent = battle.battle_tag;

      const meta = document.createElement("span");
      meta.className = "battle-meta";
      meta.textContent = `vs ${battle.opponent_kind || "unknown"} - ${battle.p1 || "p1"} + ${battle.p3 || "p3"}`;

      const result = document.createElement("span");
      const resultKind = battle.won
        ? "win"
        : battle.error
          ? "error"
          : battle.lost
            ? "loss"
            : "open";
      result.className = `battle-result ${resultKind}`;
      result.textContent = resultLabel(battle);

      summary.append(tag, meta);
      row.append(index, summary, result);
      row.addEventListener("click", () => openReplay(battle));
      fragment.append(row);
    }
    battleList.append(fragment);
  }
  return {
    clearBenchmarkStats,
    setBenchmarkResultsVisible,
    renderEmptySummary,
    renderEmptyMetrics,
    setConnectionState,
    setControlsTab,
    updateHeaderTiming,
    renderJob,
    renderLog,
    renderProgress,
    renderLiveBenchmark,
    renderSummary,
    renderBenchmarkMetrics,
    renderTabs,
    renderBattles,
  };
}
