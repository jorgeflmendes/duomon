export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "--";
  const total = Math.max(0, Math.round(Number(seconds)));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const remaining = minutes % 60;
    return `${hours}h ${remaining}m`;
  }
  if (minutes > 0) return `${minutes}m ${secs.toString().padStart(2, "0")}s`;
  return `${secs}s`;
}

export function formatTime(timestamp = Date.now()) {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function metricValue(metric) {
  if (
    !metric ||
    metric.available === false ||
    metric.value === null ||
    metric.value === undefined
  ) {
    return "N/A";
  }
  const unit = metric.unit || "";
  const value = Number(metric.value || 0);
  if (unit === "%") return `${value.toFixed(1)}%`;
  if (unit === "pts/sec") return `${value.toFixed(2)} pts/s`;
  if (unit === "pts/turn") return `${value.toFixed(2)} pts/turn`;
  return unit ? `${value.toFixed(2)} ${unit}` : value.toFixed(2);
}

export function setStatusPill(element, label, value) {
  element.replaceChildren();
  const labelNode = document.createElement("span");
  labelNode.className = "mono";
  labelNode.textContent = label;
  const valueNode = document.createElement("b");
  valueNode.className = "mono tnum";
  valueNode.textContent = value;
  element.append(labelNode, document.createTextNode(" "), valueNode);
}

export function resultLabel(battle) {
  if (battle.won) return "WIN";
  if (battle.lost) return "LOSS";
  return battle.finished ? "DONE" : battle.error ? "ERROR" : "OPEN";
}
