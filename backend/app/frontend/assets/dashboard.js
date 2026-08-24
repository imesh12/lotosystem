const yen = new Intl.NumberFormat("ja-JP", {
  style: "currency",
  currency: "JPY",
  maximumFractionDigits: 0,
});

const lotteryLabels = {
  LOTO6: "LOTO6",
  MINI_LOTO: "Mini Loto",
};

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refresh-button").addEventListener("click", loadDashboard);
  loadDashboard();
  window.setInterval(loadDashboard, 60000);
});

async function loadDashboard() {
  try {
    const response = await fetch("/api/status", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("Unable to load current system status.");
    const status = await response.json();
    renderSystem(status);
    renderLottery("LOTO6", status.LOTO6, document.getElementById("loto6-card"));
    renderLottery("MINI_LOTO", status.MINI_LOTO, document.getElementById("mini-loto-card"));
    renderFinancial(status.financial);
    renderAlerts(status);
  } catch (error) {
    document.getElementById("alert-region").innerHTML = alertHtml(error.message, true);
  }
}

function renderSystem(status) {
  const systemTime = document.getElementById("system-time");
  const automation = status.system?.automation;
  systemTime.textContent = [
    `Current JST: ${status.system?.current_jst_time ?? "unknown"}`,
    `Email: ${status.system?.email?.enabled ? "enabled" : "disabled"}`,
    `LOTO6 next: ${automation?.lotteries?.LOTO6?.next_action ?? "unknown"}`,
    `Mini Loto next: ${automation?.lotteries?.MINI_LOTO?.next_action ?? "unknown"}`,
  ].join(" · ");
}

function renderLottery(code, payload, container) {
  const latest = payload?.latest_official_draw;
  const pending = payload?.pending_prediction;
  const disabled = payload?.next_scheduled_action === "NO_ACTION" && payload?.next_run_at === null;
  container.innerHTML = `
    <h2>${lotteryLabels[code]}</h2>
    ${disabled ? '<p class="alert">Lottery automation is disabled.</p>' : ""}
    ${renderLatest(latest)}
    ${renderNextPrediction(code, pending)}
  `;
}

function renderLatest(latest) {
  if (!latest || latest.latest === null) {
    return '<section class="subsection"><h3>Official Result</h3><p class="empty">No latest result available.</p></section>';
  }
  return `
    <section class="subsection">
      <h3>Official Result</h3>
      <p><strong>#${latest.draw_number}</strong> · ${latest.draw_date}</p>
      ${numberBadges(latest.main_numbers)}
      <p>Bonus ${numberBadges(latest.bonus_numbers, "bonus")}</p>
    </section>
    <section class="subsection">
      <h3>Our Prediction For This Draw</h3>
      ${latest.prediction_available ? renderTicketResults(latest.ticket_results) : '<p class="empty">No saved prediction for this completed draw.</p>'}
      ${renderFinancialBlock(latest.paper_financial, latest.settlement_status)}
    </section>
  `;
}

function renderTicketResults(results) {
  if (!results || results.length === 0) {
    return '<p class="empty">No ticket results recorded.</p>';
  }
  return results
    .map(
      (ticket) => `
      <div class="ticket">
        <strong>Set ${ticket.ticket_index}</strong>
        ${numberBadges(ticket.numbers)}
        <div class="metric-row"><span>Matches</span><span>${ticket.main_matches}</span></div>
        <div class="metric-row"><span>Bonus matches</span><span>${ticket.bonus_matches}</span></div>
        <div class="metric-row"><span>Prize</span><span>${ticket.prize_tier ?? "No prize"}</span></div>
        <div class="metric-row"><span>Paper payout</span><span>${formatYen(ticket.payout_yen)}</span></div>
      </div>
    `,
    )
    .join("");
}

function renderFinancialBlock(financial, status) {
  if (!financial) return "";
  return `
    <div class="subsection">
      <h3>Paper Trading</h3>
      <div class="metric-row"><span>Cost</span><span>${formatYen(financial.paper_total_cost_yen)}</span></div>
      <div class="metric-row"><span>Winnings</span><span>${formatYen(financial.paper_gross_winnings_yen)}</span></div>
      <div class="metric-row"><span>Net</span><span class="${netClass(financial.paper_net_yen, status)}">${formatYen(financial.paper_net_yen, true)}</span></div>
      <span class="badge">${status ?? "No settlement"}</span>
    </div>
  `;
}

function renderNextPrediction(code, prediction) {
  if (!prediction) {
    return '<section class="subsection"><h3>Next Prediction</h3><p class="empty">No pending next prediction.</p></section>';
  }
  const tickets = prediction.tickets
    .map(
      (ticket) => `
      <div class="ticket">
        <strong>Set ${ticket.ticket_index}</strong>
        ${numberBadges(ticket.numbers)}
      </div>
    `,
    )
    .join("");
  return `
    <section class="subsection">
      <h3>Next ${lotteryLabels[code]} Prediction</h3>
      <p><strong>#${prediction.target_draw_number}</strong> · ${prediction.target_draw_date}</p>
      <p class="muted">Generated ${prediction.generated_at ?? "unknown"} · ${prediction.ticket_count} sets</p>
      ${tickets}
    </section>
  `;
}

function renderFinancial(financial) {
  const target = document.getElementById("financial-summary");
  target.innerHTML = [
    financeCard("Today", financial?.today),
    financeCard("This Month", financial?.current_month),
    financeCard("All Time", financial?.all_time),
  ].join("");
}

function financeCard(title, summary) {
  return `
    <article class="finance-card panel">
      <h3>${title}</h3>
      <div class="metric-row"><span>Paper cost</span><span>${formatYen(summary?.paper_total_cost_yen)}</span></div>
      <div class="metric-row"><span>Paper winnings</span><span>${formatYen(summary?.paper_gross_winnings_yen)}</span></div>
      <div class="metric-row"><span>Paper net</span><span class="${netClass(summary?.paper_net_yen)}">${formatYen(summary?.paper_net_yen, true)}</span></div>
    </article>
  `;
}

function renderAlerts(status) {
  const alerts = [];
  if (status.warnings) alerts.push(...status.warnings);
  const notificationFailures = status.system?.automation?.notification_failures;
  if (notificationFailures) alerts.push(`Notification failures: ${notificationFailures}`);
  document.getElementById("alert-region").innerHTML = alerts
    .map((message) => alertHtml(message, false))
    .join("");
}

function numberBadges(numbers, extraClass = "") {
  if (!numbers || numbers.length === 0) return '<span class="muted">none</span>';
  return `<div class="numbers">${numbers
    .map((number) => `<span class="ball ${extraClass}">${String(number).padStart(2, "0")}</span>`)
    .join("")}</div>`;
}

function formatYen(value, signed = false) {
  if (value === null || value === undefined) return "pending";
  if (signed && value > 0) return `+${yen.format(value)}`;
  if (signed && value < 0) return `-${yen.format(Math.abs(value))}`;
  return yen.format(value);
}

function netClass(value, status = "") {
  if (status === "PAYOUT_PENDING" || value === null || value === undefined) return "net-pending";
  if (value > 0) return "net-positive";
  if (value < 0) return "net-negative";
  return "";
}

function alertHtml(message, error) {
  return `<div class="alert ${error ? "error" : ""}">${escapeHtml(message)}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
