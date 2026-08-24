document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("settings-form").addEventListener("submit", saveSettings);
  loadSettings();
});

async function loadSettings() {
  try {
    const response = await fetch("/api/settings", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("Unable to load current settings.");
    const settings = await response.json();
    setChecked("loto6-enabled", settings.LOTO6.enabled);
    setValue("loto6-tickets", settings.LOTO6.tickets_per_draw);
    setChecked("mini-loto-enabled", settings.MINI_LOTO.enabled);
    setValue("mini-loto-tickets", settings.MINI_LOTO.tickets_per_draw);
    setChecked("email-enabled", settings.email_enabled);
  } catch (error) {
    showMessage(error.message, true);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const loto6Tickets = numberValue("loto6-tickets");
  const miniTickets = numberValue("mini-loto-tickets");
  if (!validTicketCount(loto6Tickets) || !validTicketCount(miniTickets)) {
    showMessage("Number of sets per draw must be an integer from 1 to 20.", true);
    return;
  }
  const payload = {
    LOTO6: {
      enabled: checked("loto6-enabled"),
      tickets_per_draw: loto6Tickets,
    },
    MINI_LOTO: {
      enabled: checked("mini-loto-enabled"),
      tickets_per_draw: miniTickets,
    },
    email_enabled: checked("email-enabled"),
  };
  try {
    const response = await fetch("/api/settings", {
      method: "PUT",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Unable to save settings.");
    }
    showMessage("Settings saved. Changes affect future predictions only.", false);
  } catch (error) {
    showMessage(error.message, true);
  }
}

function checked(id) {
  return document.getElementById(id).checked;
}

function setChecked(id, value) {
  document.getElementById(id).checked = Boolean(value);
}

function setValue(id, value) {
  document.getElementById(id).value = value;
}

function numberValue(id) {
  return Number.parseInt(document.getElementById(id).value, 10);
}

function validTicketCount(value) {
  return Number.isInteger(value) && value >= 1 && value <= 20;
}

function showMessage(message, error) {
  document.getElementById("settings-message").innerHTML =
    `<div class="alert ${error ? "error" : ""}">${escapeHtml(message)}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
