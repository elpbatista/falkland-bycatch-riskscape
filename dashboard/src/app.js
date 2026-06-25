const MINIMUM_FISHING_EFFORT_UNIT = 0.5;
const MIN_DISPLAY_RISK = Math.log1p(MINIMUM_FISHING_EFFORT_UNIT);
const SPECIES_USE_LOG_MIN_DISPLAY = 0.1;
const AVAILABLE_WEEKS = Array.from({ length: 52 }, (_, index) => index + 1);
const DEFAULT_WEEK = 48;
const RISK_COLORS = ["#ffffcc", "#febf5a", "#f43d25", "#800026"];
const RISK_QUANTILES = [0, 0.5, 0.9, 0.98, 1];
const STUDY_BOUNDS = [
  [-64.8, -57.5],
  [-50.2, -46.5],
];
const DATA_PATHS = {
  grid: "data/grids/h3_res6_falkland_islands.geojson",
  land: "data/falkland_islands_land.geojson",
  riskCsvRoot: "data/risk_weekly_csv",
  oceanTiles:
    "https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
};

const map = new maplibregl.Map({
  container: "map",
  center: [-57.5, -52.1],
  zoom: 4.45,
  minZoom: 3,
  maxZoom: 9,
  style: {
    version: 8,
    sources: {},
    layers: [
      {
        id: "background",
        type: "background",
        paint: { "background-color": "#eef2eb" },
      },
    ],
  },
});

map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");

const state = {
  gridLoaded: false,
  gridData: null,
  speciesCode: "BBAL",
  week: DEFAULT_WEEK,
  riskCache: new Map(),
};

const els = {
  species: document.querySelector("#species"),
  speciesOptions: document.querySelectorAll('input[name="species"]'),
  week: document.querySelector("#week"),
  weekValue: document.querySelector("#week-value"),
  previousWeek: document.querySelector("#previous-week"),
  nextWeek: document.querySelector("#next-week"),
  threshold: document.querySelector("#threshold"),
  gate: document.querySelector("#gate"),
  thresholdValue: document.querySelector("#threshold-value"),
  gateValue: document.querySelector("#gate-value"),
  classLow: document.querySelector("#class-low"),
  classMod: document.querySelector("#class-mod"),
  classHigh: document.querySelector("#class-high"),
  classXtrm: document.querySelector("#class-xtrm"),
  lowPlausibilityShare: document.querySelector("#low-plausibility-share"),
  status: document.querySelector("#status"),
};

init().catch((error) => {
  console.error(error);
  setStatus(error instanceof Error ? error.message : String(error));
});

async function init() {
  setStatus("Loading map...");
  await waitForMapLoad();

  await addGridToMap();
  setStatus(`H3 grid loaded. Loading BBAL ISO week ${state.week} risk table...`);

  bindControls();
  syncWeekControls();
  await loadActiveRisk();
  setStatus("Ready.");
}

function waitForMapLoad() {
  if (map.loaded()) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    map.once("load", resolve);
  });
}

function bindControls() {
  els.species.addEventListener("change", async (event) => {
    if (!(event.target instanceof HTMLInputElement)) return;

    setSpeciesOptionsDisabled(true);
    state.speciesCode = event.target.value;
    await loadActiveRisk();
    setSpeciesOptionsDisabled(false);
  });

  els.week.addEventListener("input", async () => {
    state.week = AVAILABLE_WEEKS[Number(els.week.value)];
    syncWeekControls();
    await loadActiveRisk();
  });

  els.previousWeek.addEventListener("click", async () => {
    await shiftWeek(-1);
  });

  els.nextWeek.addEventListener("click", async () => {
    await shiftWeek(1);
  });

  els.threshold.addEventListener("input", updateMapState);
  els.gate.addEventListener("input", updateMapState);
}

function setSpeciesOptionsDisabled(disabled) {
  for (const option of els.speciesOptions) {
    option.disabled = disabled;
  }
}

async function shiftWeek(step) {
  const currentIndex = AVAILABLE_WEEKS.indexOf(state.week);
  const nextIndex = currentIndex + step;
  if (nextIndex < 0 || nextIndex >= AVAILABLE_WEEKS.length) return;

  state.week = AVAILABLE_WEEKS[nextIndex];
  els.week.value = String(nextIndex);
  syncWeekControls();
  await loadActiveRisk();
}

function syncWeekControls() {
  const currentIndex = AVAILABLE_WEEKS.indexOf(state.week);
  els.week.value = String(currentIndex);
  els.week.max = String(AVAILABLE_WEEKS.length - 1);
  els.weekValue.textContent = `Week ${String(state.week).padStart(2, "0")}`;
  els.previousWeek.disabled = currentIndex <= 0;
  els.nextWeek.disabled = currentIndex >= AVAILABLE_WEEKS.length - 1;
}

async function addGridToMap() {
  const response = await fetch(DATA_PATHS.grid);
  if (!response.ok) {
    throw new Error(`Could not load ${DATA_PATHS.grid}: ${response.status}`);
  }

  state.gridData = await response.json();

  map.addSource("ocean-basemap", {
    type: "raster",
    tiles: [DATA_PATHS.oceanTiles],
    tileSize: 256,
    attribution: "Esri Ocean Basemap",
  });

  map.addLayer({
    id: "ocean-basemap",
    type: "raster",
    source: "ocean-basemap",
    paint: {
      "raster-opacity": 0.85,
    },
  });

  map.addSource("h3-grid", {
    type: "geojson",
    data: state.gridData,
    promoteId: "h3_index",
  });

  map.addLayer({
    id: "h3-fill",
    type: "fill",
    source: "h3-grid",
    paint: {
      "fill-color": [
        "step",
        ["coalesce", ["get", "risk"], 0.4],
        ...riskColorStops([MIN_DISPLAY_RISK, 0.8, 1.2, 1.8, 2.4]),
      ],
      "fill-opacity": [
        "case",
        ["boolean", ["get", "hasRisk"], false],
        0.9,
        0,
      ],
    },
  });

  map.addLayer({
    id: "h3-outline",
    type: "line",
    source: "h3-grid",
    paint: {
      "line-color": [
        "case",
        ["boolean", ["get", "lowPlausibility"], false],
        "rgba(92, 60, 42, 0.78)",
        "rgba(28, 54, 63, 0)",
      ],
      "line-width": [
        "case",
        ["boolean", ["get", "lowPlausibility"], false],
        0.7,
        0,
      ],
    },
  });

  map.addSource("falkland-land", {
    type: "geojson",
    data: DATA_PATHS.land,
  });

  map.addLayer({
    id: "falkland-land-fill",
    type: "fill",
    source: "falkland-land",
    paint: {
      "fill-color": "#9c9a90",
      "fill-opacity": 0.95,
    },
  });

  map.addLayer({
    id: "falkland-land-line",
    type: "line",
    source: "falkland-land",
    paint: {
      "line-color": "#5d594f",
      "line-width": 0.9,
      "line-opacity": 1,
    },
  });

  state.gridLoaded = true;
  map.fitBounds(STUDY_BOUNDS, { padding: 32, duration: 0 });
}

async function loadActiveRisk() {
  const riskRows = await loadRiskCsv(state.speciesCode);
  state.riskRows = riskRows;
  updateMapState();
}

async function loadRiskCsv(speciesCode) {
  const weekLabel = `w${String(state.week).padStart(2, "0")}`;
  const cacheKey = `${speciesCode}_${weekLabel}`;
  if (state.riskCache.has(cacheKey)) {
    return state.riskCache.get(cacheKey);
  }

  setStatus(`Loading ${speciesCode} ISO week ${state.week} risk table...`);
  const path = `${DATA_PATHS.riskCsvRoot}/${cacheKey}.csv`;
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}: ${response.status}`);
  }

  const rows = parseRiskCsv(await response.text());
  state.riskCache.set(cacheKey, rows);
  return rows;
}

function updateMapState() {
  if (!state.gridLoaded) return;

  const riskRows = state.riskRows;
  if (!riskRows) return;

  const threshold = Number(els.threshold.value);
  const gateStrength = Number(els.gate.value);
  els.thresholdValue.value = threshold.toFixed(2);
  els.gateValue.value = gateStrength.toFixed(2);

  let lowPlausibilityRepresented = 0;
  const valuesByH3 = new Map();
  const displayRisks = [];

  for (const row of riskRows) {
    const speciesUseLog = row.speciesUseLog;
    const plausibility = row.plausibility;
    const { risk, gatedSpeciesUseLog } = latentRiskValues(
      speciesUseLog,
      plausibility,
      gateStrength,
    );
    const hasRisk =
      Number.isFinite(risk) && gatedSpeciesUseLog > SPECIES_USE_LOG_MIN_DISPLAY;
    const lowPlausibility = hasRisk && plausibility < threshold;
    valuesByH3.set(row.h3, { risk, plausibility, hasRisk, lowPlausibility });

    if (hasRisk) {
      displayRisks.push(risk);
    }
    if (lowPlausibility) {
      lowPlausibilityRepresented += 1;
    }
  }

  for (const feature of state.gridData.features) {
    const h3 = BigInt(`0x${feature.properties.h3_index}`).toString();
    const values = valuesByH3.get(h3);
    feature.properties.h3 = h3;
    feature.properties.risk = values?.risk ?? null;
    feature.properties.plausibility = values?.plausibility ?? null;
    feature.properties.hasRisk = values?.hasRisk ?? false;
    feature.properties.lowPlausibility = values?.lowPlausibility ?? false;
  }
  const bins = displayRisks.length > 0 ? riskBins(displayRisks) : null;
  updateRiskColorBins(bins);
  map.getSource("h3-grid").setData(state.gridData);

  updateSummaryMetrics(displayRisks, bins, lowPlausibilityRepresented);
  setStatus(`${state.speciesCode} ISO week ${state.week}`);
}

function updateRiskColorBins(bins) {
  if (!bins) return;
  map.setPaintProperty("h3-fill", "fill-color", [
    "step",
    ["coalesce", ["get", "risk"], MIN_DISPLAY_RISK],
    ...riskColorStops(bins),
  ]);
}

function updateSummaryMetrics(displayRisks, bins, lowPlausibilityRepresented) {
  if (!bins || displayRisks.length === 0) {
    setClassShareText(["-", "-", "-", "-"]);
    els.lowPlausibilityShare.textContent = "-";
    return;
  }

  const classCounts = [0, 0, 0, 0];
  for (const risk of displayRisks) {
    classCounts[riskClassIndex(risk, bins)] += 1;
  }

  setClassShareText(
    classCounts.map((count) => count.toLocaleString()),
  );
  els.lowPlausibilityShare.textContent = formatShare(
    lowPlausibilityRepresented / displayRisks.length,
  );
}

function setClassShareText(values) {
  [els.classLow, els.classMod, els.classHigh, els.classXtrm].forEach(
    (element, index) => {
      element.textContent = values[index];
    },
  );
}

function riskClassIndex(risk, bins) {
  if (risk >= bins[3]) return 3;
  if (risk >= bins[2]) return 2;
  if (risk >= bins[1]) return 1;
  return 0;
}

function formatShare(value) {
  return `${Math.round(value * 100)}%`;
}

function riskBins(values) {
  const sorted = values.toSorted((a, b) => a - b);
  const bins = RISK_QUANTILES.map((quantile) => quantileValue(sorted, quantile));
  bins[0] = MIN_DISPLAY_RISK;

  if (bins.some((value, index) => index > 0 && value <= bins[index - 1])) {
    const upper = Math.max(quantileValue(sorted, 0.99), sorted.at(-1), MIN_DISPLAY_RISK * 1.01);
    return geometricBins(MIN_DISPLAY_RISK, upper, RISK_COLORS.length + 1);
  }

  return bins;
}

function quantileValue(sorted, quantile) {
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * quantile;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const weight = position - lowerIndex;
  return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
}

function geometricBins(lower, upper, count) {
  const bins = [];
  const ratio = upper / lower;
  for (let index = 0; index < count; index += 1) {
    bins.push(lower * ratio ** (index / (count - 1)));
  }
  return bins;
}

function riskColorStops(bins) {
  return [
    RISK_COLORS[0],
    bins[1],
    RISK_COLORS[1],
    bins[2],
    RISK_COLORS[2],
    bins[3],
    RISK_COLORS[3],
  ];
}

function parseRiskCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const rows = [];
  for (let index = 1; index < lines.length; index += 1) {
    const [h3, speciesUseLog, plausibility] = lines[index].split(",");
    rows.push({
      h3,
      speciesUseLog: Number(speciesUseLog),
      plausibility: Number(plausibility),
    });
  }
  return rows;
}

function latentRiskValues(speciesUseLogPred, plausibility, gateStrength) {
  const speciesUse = Math.max(Math.expm1(speciesUseLogPred), 0);
  const gate = 1 - gateStrength * (1 - plausibility);
  const gatedSpeciesUseLog = Math.log1p(speciesUse * gate);
  return {
    gatedSpeciesUseLog,
    risk: gatedSpeciesUseLog + Math.log1p(MINIMUM_FISHING_EFFORT_UNIT),
  };
}

function setStatus(message) {
  els.status.textContent = message;
}
