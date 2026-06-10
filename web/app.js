import { getJSON, latestJob, post } from "./dashboard/api.js";
import {
  OPPONENTS,
  OPPONENT_COLORS,
  OPPONENT_DESCRIPTIONS,
  OPPONENT_LABELS,
} from "./dashboard/constants.js";
import {
  formatDuration,
  formatTime,
  metricValue,
  resultLabel,
  setStatusPill,
} from "./dashboard/format.js";
import { createDashboardView } from "./dashboard/view.js";
const state = {
  polling: null,
  activeJob: null,
  selectedBattle: null,
  opponentFilter: "all",
  resultFilter: "all",
  metricResultFilter: "total",
  metricOpponentFilter: "all",
  summary: null,
  liveProgress: null,
  optionsDirty: false,
  suppressDirty: false,
  benchmarkResultsVisible: false,
  metricsVisible: false,
  lastLogSignature: "",
  lastBattleSignature: "",
  lastSummarySignature: "",
  lastProgressSignature: "",
  lastArtifactSignature: "",
  lastCompletedKey: "",
  lastBattleRefreshAt: 0,
  decisionTrace: null,
  traceViewMode: "current",
  currentReplayTurn: 1,
  currentReplayMaxTurn: 0,
};

const statusLine = document.querySelector("#statusLine");
const logOutput = document.querySelector("#logOutput");
const trainBtn = document.querySelector("#trainBtn");
const benchmarkBtn = document.querySelector("#benchmarkBtn");
const pauseBenchmarkBtn = document.querySelector("#pauseBenchmarkBtn");
const stopBenchmarkBtn = document.querySelector("#stopBenchmarkBtn");
const clearBtn = document.querySelector("#clearBtn");
const refreshBattlesBtn = document.querySelector("#refreshBattlesBtn");
const battleList = document.querySelector("#battleList");
const replayFrame = document.querySelector("#replayFrame");
const replayTitle = document.querySelector("#replayTitle");
const traceTitle = document.querySelector("#traceTitle");
const traceCount = document.querySelector("#traceCount");
const traceBody = document.querySelector("#traceBody");
const traceModeBtn = document.querySelector("#traceModeBtn");
const summaryTitle = document.querySelector("#summaryTitle");
const summaryCards = document.querySelector("#summaryCards");
const metricRows = document.querySelector("#metricRows");
const artifactPanel = document.querySelector("#artifactPanel");
const jobProgress = document.querySelector("#jobProgress");
const jobProgressLabel = document.querySelector("#jobProgressLabel");
const jobProgressPercent = document.querySelector("#jobProgressPercent");
const jobProgressBar = document.querySelector("#jobProgressBar");
const jobEta = document.querySelector("#jobEta");
const connectionState = document.querySelector("#connectionState");
const headerEta = document.querySelector("#headerEta");
const lastRefresh = document.querySelector("#lastRefresh");
const trainScopeLine = document.querySelector("#trainScopeLine");
const retrainNotice = document.querySelector("#retrainNotice");
const opponentTabs = document.querySelector("#opponentTabs");
const resultFilters = document.querySelector("#resultFilters");
const benchmarkTab = document.querySelector("#benchmarkTab");
const trainTab = document.querySelector("#trainTab");
const benchmarkSection = document.querySelector("#benchmarkSection");
const trainSection = document.querySelector("#trainSection");
const battleCountInput = document.querySelector("#battleCountInput");
const parallelismInput = document.querySelector("#parallelismInput");
const opponentChecks = document.querySelector("#opponentChecks");
const benchmarkProfileSelect = document.querySelector("#benchmarkProfileSelect");
const communicationToggle = document.querySelector("#communicationToggle");
const trainOpponentLabel = document.querySelector("#trainOpponentLabel");
const trainOpponentSelect = document.querySelector("#trainOpponentSelect");
const trainModeSelect = document.querySelector("#trainModeSelect");
const trainEpochsInput = document.querySelector("#trainEpochsInput");
const trainBatchInput = document.querySelector("#trainBatchInput");
const trainLrInput = document.querySelector("#trainLrInput");
const trainDropoutInput = document.querySelector("#trainDropoutInput");
const trainEarlyStopInput = document.querySelector("#trainEarlyStopInput");
const fixedTeamToggle = document.querySelector("#fixedTeamToggle");
const mirrorOpponentToggle = document.querySelector("#mirrorOpponentToggle");
const randomTeamBtn = document.querySelector("#randomTeamBtn");
const defaultTeamBtn = document.querySelector("#defaultTeamBtn");
const allyP1TeamInput = document.querySelector("#allyP1TeamInput");
const allyP3TeamInput = document.querySelector("#allyP3TeamInput");

function setStatus(text, failed = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("failed", failed);
}

function setBusy(busy) {
  trainBtn.disabled = busy;
  benchmarkBtn.disabled = busy;
  randomTeamBtn.disabled = busy;
  defaultTeamBtn.disabled = busy;
}

function updateBenchmarkControlButtons(job = null) {
  const canControl = job && job.id === "benchmark" && job.status === "running";
  pauseBenchmarkBtn.disabled = !canControl;
  stopBenchmarkBtn.disabled = !canControl;
  pauseBenchmarkBtn.textContent = job?.paused ? "Resume" : "Pause";
  pauseBenchmarkBtn.dataset.action = job?.paused ? "resume" : "pause";
}

function selectedTrainTargetLabel() {
  const value = trainOpponentSelect.value;
  if (value === "simple") return "the SimpleHeuristics CTDE model";
  if (value === "abyssal") return "the Abyssal CTDE model";
  return "the SimpleHeuristics and Abyssal CTDE models";
}

function updateTrainScope() {
  const teamScope = fixedTeamToggle.checked ? "fixed ally team" : "random ally teams";
  const opponentScope = mirrorOpponentToggle.checked
    ? "mirrored opponent teams"
    : "random opponent teams";
  const trainingMode = trainModeSelect.value === "smart" ? "smart CTDE" : "standard CTDE";
  trainScopeLine.textContent = `Current target: ${selectedTrainTargetLabel()} for ${teamScope}, ${opponentScope}, using ${trainingMode}.`;
}

function renderRetrainNotice() {
  retrainNotice.hidden = !state.optionsDirty;
}

function markOptionsDirty() {
  if (state.suppressDirty) return;
  state.optionsDirty = true;
  updateTrainScope();
  renderRetrainNotice();
}

function clearOptionsDirty() {
  state.optionsDirty = false;
  renderRetrainNotice();
}

async function refreshBattles({ final = true, reveal = true, limit = 1000 } = {}) {
  const params = new URLSearchParams({
    opponent: state.opponentFilter,
    result: state.resultFilter,
    limit: String(limit),
    turn_metrics: final ? "1" : "0",
  });
  const payload = await getJSON(`/api/battles?${params.toString()}`);
  if (reveal) {
    if (final) renderSummary(payload.summary, true);
    else if (!state.liveProgress) renderSummary(payload.summary, false);
    else renderTabs();
  }
  renderBattles(payload.battles || []);
  if (!state.selectedBattle && payload.battles && payload.battles.length) {
    const first = payload.battles.find((item) => item.replay_available);
    if (first) openReplay(first, false);
  }
}

function artifactValue(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function renderArtifacts(payload) {
  const artifacts = payload?.artifacts || payload || {};
  if (!artifactPanel) return;
  if (!artifacts.available) {
    artifactPanel.hidden = true;
    artifactPanel.replaceChildren();
    return;
  }
  const signature = JSON.stringify(artifacts);
  if (signature === state.lastArtifactSignature) return;
  state.lastArtifactSignature = signature;
  const profile = artifacts.profile || {};
  const battles = artifacts.battles || {};
  const storage = profile.storage_mib || {};
  const chips = [
    ["Run", `${artifactValue(artifacts.run_name)}/${artifactValue(artifacts.run_id)}`],
    ["Throughput", `${Number(profile.battles_per_second || 0).toFixed(2)} battles/s`],
    ["Selected traces", artifactValue(battles.selected_traces, "0")],
    ["Battle metadata", artifactValue(battles.metadata_rows, "0")],
    ["Output size", `${Number(storage.output_dir || 0).toFixed(2)} MiB`],
    ["Training size", `${Number(storage.training_dir || 0).toFixed(2)} MiB`],
    ["Seed", artifactValue(artifacts.seed)],
    ["Git", `${artifactValue(artifacts.git_commit)}${artifacts.dirty ? " dirty" : ""}`],
  ];
  artifactPanel.replaceChildren();
  for (const [label, value] of chips) {
    const chip = document.createElement("div");
    chip.className = "artifact-chip";
    const labelNode = document.createElement("span");
    labelNode.textContent = label;
    const valueNode = document.createElement("b");
    valueNode.textContent = value;
    chip.append(labelNode, valueNode);
    artifactPanel.append(chip);
  }
  artifactPanel.hidden = false;
}

async function refreshArtifacts() {
  const payload = await getJSON("/api/artifacts");
  renderArtifacts(payload);
}

function openReplay(battle, bustCache = true) {
  if (!battle.replay_available) return;
  state.selectedBattle = battle.battle_tag;
  state.decisionTrace = null;
  state.traceViewMode = "current";
  state.currentReplayTurn = 1;
  state.currentReplayMaxTurn = 0;
  replayTitle.textContent = `${battle.battle_tag} - vs ${battle.opponent_kind || "unknown"}`;
  replayFrame.src = bustCache ? `${battle.replay_url}?t=${Date.now()}` : battle.replay_url;
  state.lastBattleSignature = "";
  loadDecisionTrace(battle.battle_tag).catch((error) => {
    renderDecisionTraceError(battle.battle_tag, error.message);
  });
}

function traceValue(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

function traceNode(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function traceKV(label, value) {
  const item = traceNode("div", "trace-kv");
  item.append(traceNode("span", "", label), traceNode("b", "", traceValue(value)));
  return item;
}

function tracePill(label, value) {
  const pill = traceNode("span", "trace-pill");
  pill.textContent = `${label}: ${traceValue(value)}`;
  return pill;
}

function renderTraceList(items, formatter, emptyText) {
  const list = traceNode("div", "trace-list");
  const validItems = Array.isArray(items) ? items.filter(Boolean) : [];
  if (!validItems.length) {
    list.append(traceNode("div", "trace-item", emptyText));
    return list;
  }
  for (const item of validItems) {
    list.append(formatter(item));
  }
  return list;
}

function renderProposalItem(proposal) {
  const item = traceNode("div", "trace-item");
  const title = traceNode("strong");
  title.textContent = `${proposal.strategy || "proposal"} -> ${proposal.recommended_label || "no action"}`;
  const meta = traceNode("span");
  meta.textContent = [
    `target ${traceValue(proposal.target_species || proposal.target_slot)}`,
    `confidence ${traceValue(proposal.confidence)}`,
    `damage ${traceValue(proposal.self_damage)}`,
    proposal.solo_ko ? "solo KO" : "",
    proposal.partner_required ? "partner required" : "",
    proposal.communication_role ? `role ${proposal.communication_role}` : "",
    proposal.rationale ? `reason ${proposal.rationale}` : "",
  ]
    .filter(Boolean)
    .join(" | ");
  item.append(title, meta);
  return item;
}

function renderCandidateItem(candidate) {
  const item = traceNode("div", "trace-item");
  const title = traceNode("strong");
  title.textContent = candidate.label || candidate.signature || "candidate";
  const action = traceNode("code", "", candidate.signature || "");
  const meta = traceNode("span");
  meta.textContent = [
    `score ${traceValue(candidate.score)}`,
    candidate.base_score !== undefined ? `base ${traceValue(candidate.base_score)}` : "",
    candidate.adjustment !== undefined ? `adj ${traceValue(candidate.adjustment)}` : "",
    candidate.target ? `target ${candidate.target}` : "",
    candidate.damage_sum !== undefined ? `damage ${traceValue(candidate.damage_sum)}` : "",
    candidate.accuracy !== undefined ? `acc ${traceValue(candidate.accuracy)}` : "",
    candidate.protect ? "Protect" : "",
    candidate.tera ? "Tera" : "",
  ]
    .filter(Boolean)
    .join(" | ");
  item.append(title, action, meta);
  return item;
}

function renderRawDetails(title, payload) {
  const details = traceNode("details", "trace-details");
  const summary = traceNode("summary", "", title);
  const pre = traceNode("pre", "trace-json");
  pre.textContent = JSON.stringify(payload || {}, null, 2);
  details.append(summary, pre);
  return details;
}

function renderTraceSection(title, child) {
  const section = traceNode("div", "trace-section");
  section.append(traceNode("div", "trace-section-title", title), child);
  return section;
}

function firstCommitment(messages) {
  return (messages || []).map((message) => message.commitment).find((item) => item && item.decision);
}

function ownInformMessage(agent) {
  return (agent.messages || []).find(
    (message) => message.agent === agent.agent && message.speech_act === "inform_propose"
  );
}

function renderAgentTrace(agent) {
  const card = traceNode("article", "trace-agent");
  const head = traceNode("div", "trace-agent-head");
  head.append(
    traceNode("div", "trace-agent-name", agent.agent || "unknown agent"),
    traceNode("div", "trace-action", agent.action || "no action logged")
  );

  const selected = agent.selected || {};
  const protocol = agent.protocol || {};
  const commitment = firstCommitment(agent.messages) || {};
  const shared = commitment.shared_joint_plan || {};
  const diagnostics = agent.decision_diagnostics || {};
  const ownMessage = ownInformMessage(agent) || {};

  const kv = traceNode("div", "trace-kv-grid");
  kv.append(
    traceKV("Selected", selected.label),
    traceKV("Score", selected.score),
    traceKV("Protocol", protocol.used),
    traceKV("Protocol reason", protocol.reason),
    traceKV("Communication gain", protocol.communication_gain),
    traceKV("Agreement", protocol.message_agreement),
    traceKV("Conflict", protocol.message_conflict),
    traceKV("Plan consistent", protocol.plan_consistency)
  );

  const risk = diagnostics.risk_context || {};
  const riskGrid = traceNode("div", "trace-kv-grid");
  riskGrid.append(
    traceKV("Risk", risk.risk),
    traceKV("Targeted", risk.targeted),
    traceKV("Damage", risk.damage),
    traceKV("KO", risk.ko),
    traceKV("Primary slot", risk.primary_opp_slot),
    traceKV("Protect available", risk.protect_available)
  );

  const gates = diagnostics.selected_gates || {};
  const gateList = renderTraceList(
    Object.entries(gates),
    ([name, gate]) => {
      const item = traceNode("div", "trace-item");
      item.append(
        traceNode("strong", "", name),
        traceNode(
          "span",
          "",
          [
            gate?.reason ? `reason ${gate.reason}` : "",
            gate?.allowed !== undefined ? `allowed ${traceValue(gate.allowed)}` : "",
            gate?.adjustment !== undefined ? `adjustment ${traceValue(gate.adjustment)}` : "",
            gate?.preempts_primary_threat !== undefined
              ? `preempts threat ${traceValue(gate.preempts_primary_threat)}`
              : "",
          ]
            .filter(Boolean)
            .join(" | ")
        )
      );
      return item;
    },
    "No gate diagnostics logged."
  );

  const communication = traceNode("div", "trace-list");
  for (const message of agent.messages || []) {
    const item = traceNode("div", "trace-item");
    item.append(
      traceNode("strong", "", `${message.agent || "agent"}: ${message.speech_act || "message"}`),
      traceNode(
        "span",
        "",
        [
          message.top_strategy ? `top strategy ${message.top_strategy}` : "",
          message.commitment?.decision ? `decision ${message.commitment.decision}` : "",
          message.commitment?.reason ? `reason ${message.commitment.reason}` : "",
        ]
          .filter(Boolean)
          .join(" | ") || "structured message"
      )
    );
    communication.append(item);
  }

  const sharedGrid = traceNode("div", "trace-kv-grid");
  sharedGrid.append(
    traceKV("Joint reason", shared.reason),
    traceKV("Pair score", shared.pair_score),
    traceKV("Pair bonus", shared.pair_bonus),
    traceKV("Local action", shared.local_label),
    traceKV("Partner action", shared.partner_label),
    traceKV("Partner agent", shared.partner_agent)
  );

  card.append(
    head,
    renderTraceSection("Final decision and protocol", kv),
    renderTraceSection("Pre-communication proposals", renderTraceList(ownMessage.proposals, renderProposalItem, "No proposals logged.")),
    renderTraceSection("Communication and commitment", communication),
    renderTraceSection("Shared joint plan after communication", sharedGrid),
    renderTraceSection("Risk context", riskGrid),
    renderTraceSection("Decision gates", gateList),
    renderTraceSection("Top candidates after gates", renderTraceList(diagnostics.top_candidates, renderCandidateItem, "No candidate diagnostics logged.")),
    renderRawDetails("Raw logged data for this agent-turn", agent.raw)
  );
  return card;
}

function nearestTraceTurn(turns, requestedTurn) {
  if (!turns.length) return null;
  const wanted = Number(requestedTurn || 1);
  return (
    turns.find((turn) => Number(turn.turn) === wanted) ||
    turns.filter((turn) => Number(turn.turn) <= wanted).at(-1) ||
    turns[0]
  );
}

function visibleTraceTurns(trace) {
  const turns = Array.isArray(trace?.turns) ? trace.turns : [];
  if (state.traceViewMode === "all") return turns;
  const current = nearestTraceTurn(turns, state.currentReplayTurn);
  return current ? [current] : [];
}

function updateTraceModeButton(trace) {
  if (!traceModeBtn) return;
  const available = Boolean(trace?.available && Array.isArray(trace.turns) && trace.turns.length);
  traceModeBtn.hidden = !available;
  traceModeBtn.textContent = state.traceViewMode === "all" ? "Show current turn" : "Show all turns";
}

function renderDecisionTrace(payload = null) {
  const trace = payload ? payload.trace || payload || {} : state.decisionTrace || {};
  state.decisionTrace = trace;
  traceTitle.textContent = trace.battle_tag ? `Decision Trace - ${trace.battle_tag}` : "Decision Trace";
  updateTraceModeButton(trace);
  traceBody.replaceChildren();
  if (!trace.available) {
    traceCount.textContent = "No trace";
    const empty = traceNode("div", "empty-state");
    empty.append(
      traceNode(
        "div",
        "es-sub",
        "No structured decision trace is available for this battle. Run a benchmark with replay metrics logging enabled."
      )
    );
    traceBody.append(empty);
    return;
  }
  const turnsToRender = visibleTraceTurns(trace);
  const visibleAgentLogs = turnsToRender.reduce(
    (total, turn) => total + Number((turn.agents || []).length),
    0
  );
  const currentTurn = nearestTraceTurn(trace.turns || [], state.currentReplayTurn);
  traceCount.textContent =
    state.traceViewMode === "all"
      ? `${trace.turn_count} turns | ${trace.raw_record_count} agent-turn logs`
      : `Turn ${currentTurn?.turn || state.currentReplayTurn} | ${visibleAgentLogs} agent-turn logs`;
  const summary = traceNode("div", "trace-summary");
  if (state.traceViewMode === "all") {
    summary.append(
      tracePill("view", "all turns"),
      tracePill("battle", trace.battle_tag),
      tracePill("turns", trace.turn_count),
      tracePill("agent-turn logs", trace.raw_record_count)
    );
  } else {
    summary.append(
      tracePill("view", "current replay turn"),
      tracePill("battle", trace.battle_tag),
      tracePill("replay turn", currentTurn?.turn || state.currentReplayTurn),
      tracePill("total turns", trace.turn_count)
    );
  }
  traceBody.append(summary);
  if (!turnsToRender.length) {
    traceBody.append(
      traceNode("div", "empty-state", "No agent decision trace exists for the current replay turn.")
    );
    return;
  }
  for (const turn of turnsToRender) {
    const turnCard = traceNode("section", "trace-turn");
    const head = traceNode("div", "trace-turn-head");
    head.append(
      traceNode("div", "trace-turn-title", `Turn ${turn.turn}`),
      tracePill("agents", (turn.agents || []).length)
    );
    const grid = traceNode("div", "trace-agent-grid");
    for (const agent of turn.agents || []) {
      grid.append(renderAgentTrace(agent));
    }
    turnCard.append(head, grid);
    traceBody.append(turnCard);
  }
}

function renderDecisionTraceLoading(tag) {
  traceTitle.textContent = `Decision Trace - ${tag}`;
  traceCount.textContent = "Loading";
  traceBody.replaceChildren(traceNode("div", "empty-state", "Loading structured decision trace..."));
}

function renderDecisionTraceError(tag, message) {
  traceTitle.textContent = `Decision Trace - ${tag}`;
  traceCount.textContent = "Error";
  const empty = traceNode("div", "empty-state");
  empty.append(traceNode("div", "es-sub", message || "Failed to load decision trace."));
  traceBody.replaceChildren(empty);
}

async function loadDecisionTrace(tag) {
  if (!tag) return;
  renderDecisionTraceLoading(tag);
  const payload = await getJSON(`/api/battle_trace/${encodeURIComponent(tag)}`);
  if (state.selectedBattle !== tag) return;
  renderDecisionTrace(payload);
}

function handleReplayTurnMessage(event) {
  const data = event.data || {};
  if (!data || data.type !== "duomon:replay-turn") return;
  if (data.battle_tag && state.selectedBattle && data.battle_tag !== state.selectedBattle) return;
  const nextTurn = Number(data.turn || 1);
  const maxTurn = Number(data.max_turn || 0);
  if (!Number.isFinite(nextTurn) || nextTurn <= 0) return;
  const changed =
    nextTurn !== state.currentReplayTurn || maxTurn !== state.currentReplayMaxTurn;
  state.currentReplayTurn = nextTurn;
  state.currentReplayMaxTurn = maxTurn;
  if (changed && state.traceViewMode === "current" && state.decisionTrace?.available) {
    renderDecisionTrace();
  }
}

window.addEventListener("message", handleReplayTurnMessage);

function setOpponentFilter(opponent) {
  state.opponentFilter = opponent;
  state.lastBattleSignature = "";
  refreshBattles({ final: !state.activeJob, reveal: !state.liveProgress }).catch((error) =>
    setStatus(error.message, true)
  );
  renderTabs();
}

function schedulePoll(delay = 1000) {
  clearTimeout(state.polling);
  state.polling = setTimeout(refreshStatus, delay);
}

async function maybeRefreshRunningBattles(job) {
  if (!job || job.id !== "benchmark") return;
  const progress = job.progress || {};
  const completedKey = `${(progress.completed_opponents || []).join(",")}|${progress.current || 0}|${progress.active_opponent || ""}`;
  const now = Date.now();
  const shouldRefresh =
    completedKey !== state.lastCompletedKey || now - state.lastBattleRefreshAt > 6000;
  if (!shouldRefresh) return;
  state.lastCompletedKey = completedKey;
  state.lastBattleRefreshAt = now;
  await refreshBattles({ final: false, reveal: false, limit: 400 });
}

async function refreshStatus() {
  try {
    const payload = await getJSON("/api/status");
    setStatusPill(lastRefresh, "SYNC", formatTime());
    const job = state.activeJob ? payload.jobs[state.activeJob] : latestJob(payload);
    if (job && job.status === "running" && !state.activeJob) {
      state.activeJob = job.id;
    }
    renderJob(job);
    updateBenchmarkControlButtons(job);
    await maybeRefreshRunningBattles(job);
    if (!job || job.status !== "running") {
      clearTimeout(state.polling);
      state.polling = null;
      const completedJob = job && job.status !== "running" ? job : null;
      state.activeJob = null;
      updateBenchmarkControlButtons(null);
      if (completedJob && completedJob.id === "train" && completedJob.status === "ok") {
        clearOptionsDirty();
      }
      if (completedJob && completedJob.id === "benchmark" && completedJob.status === "ok") {
        setStatus("Benchmark finished; loading final statistics");
        await refreshBattles({ final: true, reveal: true, limit: 1000 });
        await refreshArtifacts();
        setStatus("Benchmark finished");
      }
      return job;
    }
    schedulePoll(job.id === "benchmark" ? 900 : 1200);
    return job;
  } catch (error) {
    setStatus(error.message, true);
    setConnectionState("Disconnected", "failed");
    setBusy(false);
    schedulePoll(2500);
    return null;
  }
}

function startPolling(jobId) {
  state.activeJob = jobId;
  state.lastLogSignature = "";
  clearTimeout(state.polling);
  schedulePoll(150);
}

function selectedOpponents() {
  return Array.from(opponentChecks.querySelectorAll("input[type='checkbox']:checked")).map(
    (input) => input.value
  );
}

function teamPayload() {
  return {
    fixed_ally_team_enabled: fixedTeamToggle.checked,
    mirror_opponent_team_enabled: mirrorOpponentToggle.checked,
    fixed_ally_optimize_leads_enabled: true,
    ally_p1_team: allyP1TeamInput.value,
    ally_p3_team: allyP3TeamInput.value,
  };
}

async function loadTeamDefaults() {
  state.suppressDirty = true;
  let payload;
  try {
    payload = await getJSON("/api/team");
    applyTeamPayload(payload);
  } finally {
    state.suppressDirty = false;
  }
  updateTrainScope();
  clearOptionsDirty();
}

function applyTeamPayload(payload) {
  fixedTeamToggle.checked = Boolean(payload.fixed_ally_team_enabled);
  mirrorOpponentToggle.checked = Boolean(payload.mirror_opponent_team_enabled);
  allyP1TeamInput.value = payload.ally_p1_team || "";
  allyP3TeamInput.value = payload.ally_p3_team || "";
}

async function runTrain() {
  setBusy(true);
  updateTrainScope();
  setStatus(`Starting CTDE reranker training: ${selectedTrainTargetLabel()}`);
  const payload = {
    model_type: "mlp",
    opponent: trainOpponentSelect.value,
    train_mode: trainModeSelect.value,
    epochs: trainEpochsInput.value || undefined,
    batch_size: trainBatchInput.value || undefined,
    learning_rate: trainLrInput.value || undefined,
    dropout: trainDropoutInput.value || undefined,
    early_stopping_patience: trainEarlyStopInput.value || undefined,
    ...teamPayload(),
  };
  const response = await post("/api/train", payload);
  startPolling(response.job.id);
}

async function generateRandomAllyTeam() {
  setBusy(true);
  setStatus("Generating random ally team");
  try {
    const payload = await post("/api/team/random", { persist: false });
    state.suppressDirty = true;
    applyTeamPayload(payload);
    state.suppressDirty = false;
    markOptionsDirty();
    setStatus("Random ally team generated");
  } finally {
    state.suppressDirty = false;
    setBusy(false);
  }
}

async function restoreCuratedTeam() {
  setBusy(true);
  setStatus("Restoring curated ally team");
  try {
    const payload = await post("/api/team/default", {});
    state.suppressDirty = true;
    applyTeamPayload(payload);
    state.suppressDirty = false;
    markOptionsDirty();
    setStatus("Curated ally team restored");
  } finally {
    state.suppressDirty = false;
    setBusy(false);
  }
}

async function runBenchmark() {
  const opponents = selectedOpponents();
  if (!opponents.length) {
    setStatus("Select at least one opponent", true);
    return;
  }
  setBusy(true);
  setBenchmarkResultsVisible(false);
  state.selectedBattle = null;
  state.decisionTrace = null;
  state.traceViewMode = "current";
  state.currentReplayTurn = 1;
  state.currentReplayMaxTurn = 0;
  state.lastBattleSignature = "";
  state.lastCompletedKey = "";
  battleList.replaceChildren();
  replayFrame.src = "about:blank";
  replayTitle.textContent = "Replay Viewer";
  renderDecisionTrace({ available: false });
  logOutput.textContent = "Starting benchmark...";
  setStatus("Starting parallel benchmark");
  const response = await post("/api/benchmark", {
    battles: battleCountInput.value || 200,
    parallelism: parallelismInput.value || 32,
    opponents,
    profile: benchmarkProfileSelect ? benchmarkProfileSelect.value : "ctde_mlp",
    communication_enabled: communicationToggle ? communicationToggle.checked : true,
    ...teamPayload(),
  });
  renderJob(response.job);
  updateBenchmarkControlButtons(response.job);
  refreshArtifacts().catch(() => {});
  startPolling(response.job.id);
}

async function toggleBenchmarkPause() {
  const action = pauseBenchmarkBtn.dataset.action === "resume" ? "resume" : "pause";
  const response = await post(`/api/benchmark/${action}`, {});
  renderJob(response.job);
  updateBenchmarkControlButtons(response.job);
  startPolling("benchmark");
  setStatus(action === "pause" ? "Benchmark pause requested" : "Benchmark resumed");
}

async function stopBenchmark() {
  const response = await post("/api/benchmark/stop", {});
  renderJob(response.job);
  updateBenchmarkControlButtons(response.job);
  startPolling("benchmark");
  setStatus("Stopping benchmark");
}

const dashboardView = createDashboardView({
  state,
  constants: { OPPONENTS, OPPONENT_LABELS, OPPONENT_COLORS, OPPONENT_DESCRIPTIONS },
  elements: {
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
  },
  helpers: { formatDuration, formatTime, metricValue, resultLabel, setStatusPill },
  actions: { openReplay, setOpponentFilter, setBusy, setStatus },
});

const {
  clearBenchmarkStats,
  setBenchmarkResultsVisible,
  setConnectionState,
  setControlsTab,
  updateHeaderTiming,
  renderJob,
  renderSummary,
  renderTabs,
  renderBattles,
} = dashboardView;
trainBtn.addEventListener("click", () =>
  runTrain().catch((error) => {
    setStatus(error.message, true);
    setBusy(false);
  })
);

benchmarkBtn.addEventListener("click", () =>
  runBenchmark().catch((error) => {
    setStatus(error.message, true);
    setBusy(false);
    updateBenchmarkControlButtons(null);
  })
);

pauseBenchmarkBtn.addEventListener("click", () =>
  toggleBenchmarkPause().catch((error) => {
    setStatus(error.message, true);
  })
);

stopBenchmarkBtn.addEventListener("click", () =>
  stopBenchmark().catch((error) => {
    setStatus(error.message, true);
  })
);

randomTeamBtn.addEventListener("click", () =>
  generateRandomAllyTeam().catch((error) => {
    setStatus(error.message, true);
    setBusy(false);
  })
);

defaultTeamBtn.addEventListener("click", () =>
  restoreCuratedTeam().catch((error) => {
    setStatus(error.message, true);
    setBusy(false);
  })
);

benchmarkTab.addEventListener("click", () => setControlsTab("benchmark"));
trainTab.addEventListener("click", () => setControlsTab("train"));

traceModeBtn.addEventListener("click", () => {
  state.traceViewMode = state.traceViewMode === "all" ? "current" : "all";
  renderDecisionTrace();
});

refreshBattlesBtn.addEventListener("click", () =>
  Promise.all([refreshBattles({ final: true, reveal: true }), refreshArtifacts()]).catch(
    (error) => {
      setStatus(error.message, true);
    }
  )
);

resultFilters.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-result]");
  if (!button) return;
  state.resultFilter = button.dataset.result;
  state.lastBattleSignature = "";
  for (const item of resultFilters.querySelectorAll("button")) {
    item.classList.toggle("active", item === button);
  }
  refreshBattles({ final: !state.activeJob, reveal: !state.liveProgress }).catch((error) =>
    setStatus(error.message, true)
  );
});

clearBtn.addEventListener("click", () => {
  logOutput.textContent = "No job running.";
  state.lastLogSignature = "";
});

fixedTeamToggle.addEventListener("change", () => {
  if (!fixedTeamToggle.checked) {
    mirrorOpponentToggle.checked = false;
  }
  updateTrainScope();
});

mirrorOpponentToggle.addEventListener("change", () => {
  if (mirrorOpponentToggle.checked) {
    fixedTeamToggle.checked = true;
  }
  updateTrainScope();
});

[
  trainOpponentSelect,
  trainModeSelect,
  trainEpochsInput,
  trainBatchInput,
  trainLrInput,
  trainDropoutInput,
  trainEarlyStopInput,
  fixedTeamToggle,
  mirrorOpponentToggle,
  allyP1TeamInput,
  allyP3TeamInput,
].forEach((control) => {
  control.addEventListener("input", markOptionsDirty);
  control.addEventListener("change", markOptionsDirty);
});

setControlsTab("benchmark");

loadTeamDefaults()
  .catch((error) => setStatus(error.message, true))
  .finally(async () => {
    setBenchmarkResultsVisible(false);
    setConnectionState("Ready", "idle");
    updateHeaderTiming(null);
    await refreshStatus();
    await refreshArtifacts().catch(() => {});
    if (!state.activeJob) {
      await refreshBattles({ final: true, reveal: true, limit: 1000 });
    }
  });
