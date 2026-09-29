// SmartDine Real-Time Client Application Logic

const API_BASE = window.location.origin;

// State
let appMetadata = null;
let currentPresets = [];
let activePresetId = "G001";
let currentRecommendations = [];
let currentGroupSummary = null;

// Initial Default Members (Matches Benchmark G001)
let groupMembers = [
  {
    name: "Ananya",
    preferred_cuisines: ["healthy food", "sandwich"],
    budget: 600,
    dietary_preference: "vegan",
    preferred_location: "electronic city",
    preferred_restaurant_type: "cafe",
    cuisine_importance: 1.0,
    budget_importance: 1.0,
    dietary_importance: 1.0
  },
  {
    name: "Rahul",
    preferred_cuisines: ["burger", "fast food"],
    budget: 800,
    dietary_preference: "non-vegetarian",
    preferred_location: "electronic city",
    preferred_restaurant_type: "food court",
    cuisine_importance: 1.0,
    budget_importance: 1.0,
    dietary_importance: 1.0
  },
  {
    name: "Sneha",
    preferred_cuisines: ["healthy food", "salad"],
    budget: 700,
    dietary_preference: "vegetarian",
    preferred_location: "malleshwaram",
    preferred_restaurant_type: "cafe",
    cuisine_importance: 1.0,
    budget_importance: 1.0,
    dietary_importance: 1.0
  },
  {
    name: "Vikram",
    preferred_cuisines: ["seafood", "fast food"],
    budget: 900,
    dietary_preference: "non-vegetarian",
    preferred_location: "bannerghatta road",
    preferred_restaurant_type: "bar",
    cuisine_importance: 1.0,
    budget_importance: 1.0,
    dietary_importance: 1.0
  },
  {
    name: "Pooja",
    preferred_cuisines: ["healthy food", "dessert"],
    budget: 650,
    dietary_preference: "vegetarian",
    preferred_location: "electronic city",
    preferred_restaurant_type: "dessert parlor",
    cuisine_importance: 1.0,
    budget_importance: 1.0,
    dietary_importance: 1.0
  }
];

const AVATAR_COLORS = [
  "linear-gradient(135deg, #6366f1, #8b5cf6)",
  "linear-gradient(135deg, #10b981, #059669)",
  "linear-gradient(135deg, #f59e0b, #d97706)",
  "linear-gradient(135deg, #ec4899, #be185d)",
  "linear-gradient(135deg, #3b82f6, #1d4ed8)",
  "linear-gradient(135deg, #06b6d4, #0891b2)"
];

// ------------------------------------------------------------
// Initialization
// ------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  await loadMetadata();
  await loadPresets();
  renderMembers();
  updateConsensusMeter();
  await triggerRecommendation();
});

function setupEventListeners() {
  document.getElementById("addMemberBtn").addEventListener("click", addNewMember);
  document.getElementById("recommendBtn").addEventListener("click", triggerRecommendation);
  
  // Tab Switcher
  document.getElementById("tabRecommendations").addEventListener("click", () => switchTab("recs"));
  document.getElementById("tabModelExplain").addEventListener("click", () => switchTab("explain"));
  
  // Filters trigger refresh
  document.getElementById("locationFilter").addEventListener("change", triggerRecommendation);
  document.getElementById("cuisineFilter").addEventListener("change", triggerRecommendation);
  document.getElementById("minRatingFilter").addEventListener("change", triggerRecommendation);
  document.getElementById("maxBudgetFilter").addEventListener("change", triggerRecommendation);
  document.getElementById("bookingToggle").addEventListener("change", triggerRecommendation);
  document.getElementById("onlineOrderToggle").addEventListener("change", triggerRecommendation);

  // Modal close
  document.getElementById("closeModalBtn").addEventListener("click", () => {
    document.getElementById("detailModal").classList.add("hidden");
  });
  document.getElementById("detailModal").addEventListener("click", (e) => {
    if (e.target.id === "detailModal") {
      document.getElementById("detailModal").classList.add("hidden");
    }
  });
}

function switchTab(tab) {
  const recTabBtn = document.getElementById("tabRecommendations");
  const expTabBtn = document.getElementById("tabModelExplain");
  const recsContainer = document.getElementById("recommendationsList");
  const insightsContainer = document.getElementById("modelInsightsTab");

  if (tab === "recs") {
    recTabBtn.classList.add("active");
    expTabBtn.classList.remove("active");
    recsContainer.classList.remove("hidden");
    insightsContainer.classList.add("hidden");
  } else {
    expTabBtn.classList.add("active");
    recTabBtn.classList.remove("active");
    recsContainer.classList.add("hidden");
    insightsContainer.classList.remove("hidden");
  }
}

// ------------------------------------------------------------
// Metadata & Presets Loading
// ------------------------------------------------------------
async function loadMetadata() {
  try {
    const res = await fetch(`${API_BASE}/api/metadata`);
    appMetadata = await res.json();

    // Populate Location Filter Dropdown
    const locSelect = document.getElementById("locationFilter");
    locSelect.innerHTML = '<option value="">All Bangalore (12,499 Venues)</option>';
    appMetadata.locations.forEach(loc => {
      const opt = document.createElement("option");
      opt.value = loc;
      opt.textContent = capitalize(loc);
      locSelect.appendChild(opt);
    });

    // Populate Cuisine Filter Dropdown
    const cuiSelect = document.getElementById("cuisineFilter");
    cuiSelect.innerHTML = '<option value="">Any Cuisine Match</option>';
    appMetadata.cuisines.slice(0, 30).forEach(c => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = capitalize(c);
      cuiSelect.appendChild(opt);
    });

    // Populate Insights View
    renderModelInsights();
  } catch (err) {
    console.error("Failed to load metadata:", err);
  }
}

async function loadPresets() {
  try {
    const res = await fetch(`${API_BASE}/api/presets`);
    currentPresets = await res.json();

    const container = document.getElementById("presetButtonsContainer");
    container.innerHTML = "";

    currentPresets.forEach((preset, idx) => {
      const btn = document.createElement("button");
      btn.className = `preset-btn ${preset.id === activePresetId ? "active" : ""}`;
      btn.textContent = preset.id;
      btn.title = preset.description;
      btn.addEventListener("click", () => {
        applyPreset(preset);
      });
      container.appendChild(btn);
    });
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function applyPreset(preset) {
  activePresetId = preset.id;
  groupMembers = JSON.parse(JSON.stringify(preset.members));
  
  // Highlight active button
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.classList.toggle("active", btn.textContent === preset.id);
  });

  renderMembers();
  updateConsensusMeter();
  triggerRecommendation();
}

// ------------------------------------------------------------
// Group Members Management
// ------------------------------------------------------------
function renderMembers() {
  const container = document.getElementById("membersContainer");
  container.innerHTML = "";

  groupMembers.forEach((member, index) => {
    const card = document.createElement("div");
    card.className = "member-card";

    const color = AVATAR_COLORS[index % AVATAR_COLORS.length];
    const initial = (member.name || "M").charAt(0).toUpperCase();

    card.innerHTML = `
      <div class="member-card-header">
        <div class="member-avatar-name">
          <div class="member-avatar" style="background: ${color};">${initial}</div>
          <input type="text" class="member-name-input" value="${escapeHtml(member.name)}" placeholder="Member Name" data-index="${index}">
        </div>
        ${groupMembers.length > 1 ? `
          <button class="remove-member-btn" title="Remove Member" data-index="${index}">&times;</button>
        ` : ''}
      </div>

      <div class="diet-selector">
        <div class="diet-pill ${member.dietary_preference === 'vegetarian' ? 'selected' : ''}" data-index="${index}" data-diet="vegetarian">🌱 Veg</div>
        <div class="diet-pill ${member.dietary_preference === 'vegan' ? 'selected' : ''}" data-index="${index}" data-diet="vegan">🌿 Vegan</div>
        <div class="diet-pill ${member.dietary_preference === 'non-vegetarian' ? 'selected' : ''}" data-index="${index}" data-diet="non-vegetarian">🍗 Non-Veg</div>
        <div class="diet-pill ${!member.dietary_preference || member.dietary_preference === 'no preference' ? 'selected' : ''}" data-index="${index}" data-diet="no preference">Any</div>
      </div>

      <div class="member-field-row">
        <span class="member-field-label">Budget for Two:</span>
        <span class="member-budget-val" id="budgetDisplay_${index}">₹${member.budget}</span>
      </div>
      <input type="range" class="member-range-slider" min="200" max="2500" step="50" value="${member.budget}" data-index="${index}">

      <div class="member-dropdowns">
        <select class="member-mini-select loc-select" data-index="${index}">
          <option value="">Preferred Location</option>
          ${getLocationOptions(member.preferred_location)}
        </select>
        <select class="member-mini-select type-select" data-index="${index}">
          <option value="">Dining Style</option>
          ${getTypeOptions(member.preferred_restaurant_type)}
        </select>
      </div>

      <div class="member-cuisines-tags">
        ${member.preferred_cuisines.map(c => `<span class="cuisine-tag">${capitalize(c)}</span>`).join('')}
      </div>
    `;

    container.appendChild(card);
  });

  // Attach card event listeners
  container.querySelectorAll(".member-name-input").forEach(input => {
    input.addEventListener("input", (e) => {
      const idx = parseInt(e.target.dataset.index);
      groupMembers[idx].name = e.target.value;
      updateConsensusMeter();
    });
  });

  container.querySelectorAll(".remove-member-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.target.dataset.index);
      groupMembers.splice(idx, 1);
      renderMembers();
      updateConsensusMeter();
      triggerRecommendation();
    });
  });

  container.querySelectorAll(".diet-pill").forEach(pill => {
    pill.addEventListener("click", (e) => {
      const idx = parseInt(e.target.dataset.index);
      const diet = e.target.dataset.diet;
      groupMembers[idx].dietary_preference = diet;
      renderMembers();
      updateConsensusMeter();
      triggerRecommendation();
    });
  });

  container.querySelectorAll(".member-range-slider").forEach(slider => {
    slider.addEventListener("input", (e) => {
      const idx = parseInt(e.target.dataset.index);
      const val = parseInt(e.target.value);
      groupMembers[idx].budget = val;
      const display = document.getElementById(`budgetDisplay_${idx}`);
      if (display) display.textContent = `₹${val}`;
      updateConsensusMeter();
    });
    slider.addEventListener("change", triggerRecommendation);
  });

  container.querySelectorAll(".loc-select").forEach(sel => {
    sel.addEventListener("change", (e) => {
      const idx = parseInt(e.target.dataset.index);
      groupMembers[idx].preferred_location = e.target.value;
      updateConsensusMeter();
      triggerRecommendation();
    });
  });

  container.querySelectorAll(".type-select").forEach(sel => {
    sel.addEventListener("change", (e) => {
      const idx = parseInt(e.target.dataset.index);
      groupMembers[idx].preferred_restaurant_type = e.target.value;
      updateConsensusMeter();
      triggerRecommendation();
    });
  });
}

function getLocationOptions(selected) {
  if (!appMetadata) return "";
  return appMetadata.locations.map(loc => {
    const isSel = selected && loc.toLowerCase() === selected.toLowerCase() ? "selected" : "";
    return `<option value="${loc}" ${isSel}>${capitalize(loc)}</option>`;
  }).join('');
}

function getTypeOptions(selected) {
  if (!appMetadata) return "";
  return appMetadata.restaurant_types.map(t => {
    const isSel = selected && t.toLowerCase() === selected.toLowerCase() ? "selected" : "";
    return `<option value="${t}" ${isSel}>${capitalize(t)}</option>`;
  }).join('');
}

function addNewMember() {
  const newIdx = groupMembers.length + 1;
  groupMembers.push({
    name: `Friend ${newIdx}`,
    preferred_cuisines: ["north indian", "fast food"],
    budget: 700,
    dietary_preference: "no preference",
    preferred_location: "koramangala 5th block",
    preferred_restaurant_type: "casual dining",
    cuisine_importance: 1.0,
    budget_importance: 1.0,
    dietary_importance: 1.0
  });

  renderMembers();
  updateConsensusMeter();
  triggerRecommendation();
}

// ------------------------------------------------------------
// Live Consensus Calculation
// ------------------------------------------------------------
function updateConsensusMeter() {
  if (groupMembers.length === 0) return;

  const agreementDims = [];
  
  // Cuisines
  const allCuisines = groupMembers.flatMap(m => m.preferred_cuisines || []);
  if (allCuisines.length > 0) {
    const cCounts = {};
    allCuisines.forEach(c => cCounts[c] = (cCounts[c] || 0) + 1);
    const maxC = Math.max(...Object.values(cCounts));
    agreementDims.push(maxC / allCuisines.length);
  }

  // Diets
  const allDiets = groupMembers.map(m => m.dietary_preference || "any");
  const dCounts = {};
  allDiets.forEach(d => dCounts[d] = (dCounts[d] || 0) + 1);
  agreementDims.push(Math.max(...Object.values(dCounts)) / allDiets.length);

  // Locations
  const allLocs = groupMembers.map(m => m.preferred_location || "any");
  const lCounts = {};
  allLocs.forEach(l => lCounts[l] = (lCounts[l] || 0) + 1);
  agreementDims.push(Math.max(...Object.values(lCounts)) / allLocs.length);

  // Budget Variance
  const budgets = groupMembers.map(m => m.budget);
  const minB = Math.min(...budgets);
  const maxB = Math.max(...budgets);
  const bVarianceScore = maxB > 0 ? Math.max(0, 1 - ((maxB - minB) / maxB)) : 1.0;
  agreementDims.push(bVarianceScore);

  const avgAgreement = agreementDims.reduce((a, b) => a + b, 0) / agreementDims.length;
  const pct = Math.round(avgAgreement * 100);

  const textEl = document.getElementById("consensusScoreText");
  const barEl = document.getElementById("consensusBar");
  const descEl = document.getElementById("consensusDescription");

  if (textEl && barEl) {
    textEl.textContent = `${pct}%`;
    barEl.style.width = `${pct}%`;

    if (pct >= 75) {
      descEl.textContent = "High Group Consensus: Members share very similar cuisines, diets, and budget targets.";
      barEl.style.background = "linear-gradient(90deg, #10b981, #059669)";
    } else if (pct >= 50) {
      descEl.textContent = "Moderate Harmony: Balanced preferences; ML algorithm will compromise fairly across tastes.";
      barEl.style.background = "linear-gradient(90deg, #10b981, #f59e0b)";
    } else {
      descEl.textContent = "Divergent Preferences: High conflict in dietary or budget constraints; fair multi-criteria scoring active.";
      barEl.style.background = "linear-gradient(90deg, #f59e0b, #ef4444)";
    }
  }
}

// ------------------------------------------------------------
// Recommendation Scoring & Rendering
// ------------------------------------------------------------
async function triggerRecommendation() {
  const container = document.getElementById("recommendationsList");
  container.innerHTML = `
    <div class="loading-state">
      <div class="spinner"></div>
      <p>Scoring 12,499 restaurants with trained Linear Regression model...</p>
    </div>
  `;

  const payload = {
    members: groupMembers,
    top_n: 10,
    filter_location: document.getElementById("locationFilter").value || null,
    filter_cuisine: document.getElementById("cuisineFilter").value || null,
    min_rating: document.getElementById("minRatingFilter").value ? parseFloat(document.getElementById("minRatingFilter").value) : null,
    max_budget_limit: document.getElementById("maxBudgetFilter").value ? parseFloat(document.getElementById("maxBudgetFilter").value) : null,
    require_booking: document.getElementById("bookingToggle").checked || null,
    require_online: document.getElementById("onlineOrderToggle").checked || null
  };

  const startTime = performance.now();

  try {
    const res = await fetch(`${API_BASE}/api/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    const duration = Math.round(performance.now() - startTime);

    document.getElementById("latencyText").textContent = `Latency: ${duration}ms`;
    currentRecommendations = data.recommendations;
    currentGroupSummary = data.group_summary;

    renderRecommendations(data);
  } catch (err) {
    console.error("Recommendation error:", err);
    container.innerHTML = `
      <div class="loading-state">
        <p style="color: var(--accent-rose)">Failed to fetch recommendations. Ensure the server is running.</p>
      </div>
    `;
  }
}

function renderRecommendations(data) {
  const container = document.getElementById("recommendationsList");
  const summary = data.group_summary;
  const items = data.recommendations;

  // Update summary strip
  document.getElementById("statGroupSize").textContent = `${summary.group_size} Members`;
  document.getElementById("statBudget").textContent = `₹${summary.weighted_budget}`;
  document.getElementById("statDietary").textContent = summary.dietary_mix.map(capitalize).join(", ") || "None";
  document.getElementById("statCuisines").textContent = summary.top_cuisines.map(capitalize).join(", ") || "Diverse";
  document.getElementById("resultsCountBadge").textContent = `${items.length} Matches Found`;

  if (items.length === 0) {
    container.innerHTML = `
      <div class="loading-state">
        <p>No restaurants matched all active strict filters. Try clearing some constraints.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = "";

  items.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "rec-card";

    const isTopPick = idx === 0;
    const ratingDisplay = item.rating ? `★ ${item.rating.toFixed(1)} / 5` : "★ New";
    const cuisinesList = item.cuisines ? item.cuisines.split(",").slice(0, 4) : [];

    card.innerHTML = `
      <div class="rec-card-top">
        <div class="rec-meta">
          <div class="rec-rank-name">
            <span class="rank-badge ${isTopPick ? 'top-pick' : ''}">${isTopPick ? '🏆 #1 Top Match' : `#${item.rank}`}</span>
            <h3 class="rec-name">${escapeHtml(item.name)}</h3>
          </div>
          <div class="rec-location-type">
            <span><span class="loc-pin">📍</span> ${escapeHtml(item.location)}</span>
            <span>•</span>
            <span>${escapeHtml(item.rest_type)}</span>
          </div>
          <div class="rec-cuisines">
            ${cuisinesList.map(c => `<span class="c-pill">${capitalize(c.trim())}</span>`).join('')}
          </div>
        </div>

        <div class="rec-score-box">
          <div class="score-circle-val">${item.match_percentage}%</div>
          <div class="score-label">Suitability Match</div>
        </div>
      </div>

      <div class="rec-specs-row">
        <div class="spec-chip">
          <span>Cost:</span>
          <span class="spec-val">₹${item.cost_for_two} for two</span>
        </div>
        <div class="spec-chip">
          <span class="star-gold">★</span>
          <span class="spec-val">${ratingDisplay}</span>
          <span>(${item.votes} votes)</span>
        </div>
        ${item.book_table ? `<div class="spec-chip"><span style="color: #34d399">✓</span> Table Booking</div>` : ''}
        ${item.online_order ? `<div class="spec-chip"><span style="color: #38bdf8">✓</span> Online Ordering</div>` : ''}
      </div>

      <div class="rec-explanation-box">
        <div class="exp-heading">SmartDine AI Rationale</div>
        <div class="exp-text">${escapeHtml(item.explanation)}</div>
      </div>

      <div class="fit-bars-grid">
        <div class="fit-bar-item">
          <div class="fit-bar-header">
            <span class="fit-bar-name">Cuisine Match</span>
            <span class="fit-bar-pct">${item.breakdown.cuisine_match}%</span>
          </div>
          <div class="mini-progress-bg">
            <div class="mini-progress-fill" style="width: ${item.breakdown.cuisine_match}%;"></div>
          </div>
        </div>

        <div class="fit-bar-item">
          <div class="fit-bar-header">
            <span class="fit-bar-name">Budget Fit</span>
            <span class="fit-bar-pct">${item.breakdown.budget_match}%</span>
          </div>
          <div class="mini-progress-bg">
            <div class="mini-progress-fill" style="width: ${item.breakdown.budget_match}%; background: var(--accent-amber);"></div>
          </div>
        </div>

        <div class="fit-bar-item">
          <div class="fit-bar-header">
            <span class="fit-bar-name">Dietary Safety</span>
            <span class="fit-bar-pct">${item.breakdown.dietary_compatibility}%</span>
          </div>
          <div class="mini-progress-bg">
            <div class="mini-progress-fill" style="width: ${item.breakdown.dietary_compatibility}%; background: var(--accent-emerald);"></div>
          </div>
        </div>

        <div class="fit-bar-item">
          <div class="fit-bar-header">
            <span class="fit-bar-name">Location Match</span>
            <span class="fit-bar-pct">${item.breakdown.location_match}%</span>
          </div>
          <div class="mini-progress-bg">
            <div class="mini-progress-fill" style="width: ${item.breakdown.location_match}%; background: var(--accent-purple);"></div>
          </div>
        </div>
      </div>

      <div class="rec-actions">
        <button class="btn-outline inspect-member-btn" data-index="${idx}">
          Inspect Member Satisfaction
        </button>
      </div>
    `;

    container.appendChild(card);
  });

  // Modal inspection triggers
  container.querySelectorAll(".inspect-member-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.target.dataset.index);
      openMemberDeepDive(items[idx]);
    });
  });
}

// ------------------------------------------------------------
// Member Deep Dive Modal
// ------------------------------------------------------------
function openMemberDeepDive(restaurant) {
  document.getElementById("modalRestaurantName").textContent = `${restaurant.name} (${restaurant.location})`;
  const body = document.getElementById("modalBody");

  let html = `
    <div style="margin-bottom: 1.5rem;">
      <p style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.5;">
        Here is how this venue accommodates each specific group member's constraints:
      </p>
    </div>
    <div style="display: flex; flex-direction: column; gap: 1rem;">
  `;

  groupMembers.forEach((member, i) => {
    const isVegReq = member.dietary_preference === "vegetarian" || member.dietary_preference === "vegan";
    const budgetRatio = restaurant.cost_for_two / Math.max(1, member.budget);
    let budgetStatus = "Within budget";
    let budgetColor = "var(--accent-emerald)";
    if (budgetRatio > 1.25) {
      budgetStatus = `+${Math.round((budgetRatio - 1) * 100)}% over budget`;
      budgetColor = "var(--accent-rose)";
    }

    html += `
      <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
          <strong style="color: var(--text-main); font-size: 0.95rem;">${escapeHtml(member.name)}</strong>
          <span style="font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: var(--radius-pill); background: rgba(255, 255, 255, 0.05); color: var(--text-muted);">${capitalize(member.dietary_preference || "Any diet")}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.8rem; color: var(--text-muted);">
          <div>Target Budget: <span style="color: #fff">₹${member.budget}</span> (<span style="color: ${budgetColor}">${budgetStatus}</span>)</div>
          <div>Location Preference: <span style="color: #fff">${capitalize(member.preferred_location || "Flexible")}</span></div>
        </div>
      </div>
    `;
  });

  html += `</div>`;
  body.innerHTML = html;
  document.getElementById("detailModal").classList.remove("hidden");
}

// ------------------------------------------------------------
// Model Insights Tab Rendering
// ------------------------------------------------------------
function renderModelInsights() {
  if (!appMetadata) return;

  const weightsContainer = document.getElementById("weightsGrid");
  const coeffs = appMetadata.feature_coefficients;
  
  if (coeffs && Object.keys(coeffs).length > 0) {
    weightsContainer.innerHTML = "";
    
    // Sort features by absolute coefficient
    const sorted = Object.entries(coeffs).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const maxVal = Math.max(...sorted.map(s => Math.abs(s[1])));

    sorted.slice(0, 10).forEach(([feature, val]) => {
      const pct = Math.round((Math.abs(val) / maxVal) * 100);
      const row = document.createElement("div");
      row.className = "weight-row";
      row.innerHTML = `
        <span class="weight-name">${formatFeatureName(feature)}</span>
        <div class="weight-bar-bg">
          <div class="weight-bar-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="weight-val">${val >= 0 ? '+' : ''}${val.toFixed(4)}</span>
      `;
      weightsContainer.appendChild(row);
    });
  }

  // Populate benchmark table
  const tbody = document.getElementById("benchmarkTableBody");
  if (tbody && appMetadata.model_comparison) {
    tbody.innerHTML = "";
    appMetadata.model_comparison.forEach(m => {
      const isChamp = m.Model === "Linear Regression";
      const tr = document.createElement("tr");
      if (isChamp) tr.className = "champion-row";
      tr.innerHTML = `
        <td>${isChamp ? '🏆 ' : ''}${m.Model}</td>
        <td>${m.MAE.toFixed(4)}</td>
        <td>${m.RMSE.toFixed(4)}</td>
        <td><strong>${m.R2.toFixed(4)}</strong></td>
        <td>${m.Training_Time_Seconds.toFixed(2)}s</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

// ------------------------------------------------------------
// Utility Functions
// ------------------------------------------------------------
function capitalize(str) {
  if (!str) return "";
  return str.split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[m]);
}

function formatFeatureName(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
}
