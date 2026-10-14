// App.jsx — root of the KAAL tactical dashboard.
// Owns the run/stop state, the (cosmetic) model label and the live settings.
// Everything else is derived from the latest backend snapshot delivered by
// useDefenseSocket (WebSocket /ws, REST polling fallback): video feed,
// detections, alert state (audio beep on alert onset), graphs and system
// metrics. Threshold / confidence changes go to PUT /api/settings, with an
// X-KAAL-Token header when VITE_KAAL_API_TOKEN is set.
import { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import "./App.css";
import MainPanel from "./components/MainPanel";
import GraphPanel from "./components/GraphPanel";
import MultiGraph from "./components/MultiGraph";
import Controls from "./components/Controls";
import HealthMonitor from "./components/SystemStatus";
import StatusBar from "./components/StatusBar";
import SettingsPanel from "./components/SettingsPanel";
import useDefenseSocket from "./lib/useDefenseSocket";

const VIDEO_FEED = "/api/video_feed";
const API_TOKEN = import.meta.env.VITE_KAAL_API_TOKEN;
const EMPTY_METRICS = {
  cpu: 0,
  memory: 0,
  gpu: null,
  bandwidth: 0,
  latency: 0,
  uptime: "00:00:00",
};

function App() {
  const [selectedModel, setSelectedModel] = useState("Model 1");
  const [isRunning, setIsRunning] = useState(true);
  const [settings, setSettings] = useState({ fft_threshold: null, conf: 0.4 });
  // Swarm telemetry comes from the swarm canvas (alive / kia / mode); the
  // defense snapshot only contributes status + noise below.
  const [swarmTelemetry, setSwarmTelemetry] = useState({
    status: "NOMINAL",
    alive: 0,
    kia: 0,
    mode: "NORMAL",
  });

  // Real-time stream (WebSocket with polling fallback).
  const { snapshot, connected } = useDefenseSocket(isRunning ? 1000 : null);

  // Everything below is derived from the latest snapshot — no state copies.
  const data = snapshot?.data ?? [];
  const systemMetrics = snapshot?.systemMetrics ?? EMPTY_METRICS;
  const detections = snapshot?.detections ?? [];
  const attack = Boolean(snapshot?.is_attack);
  const flaggedCount = snapshot?.flagged_count ?? 0;
  const temporal = snapshot?.temporal ?? null;
  const detector = snapshot?.detector ?? null;
  const operatingMode = snapshot?.operating_mode ?? "INITIALIZING";
  const feedStatus = snapshot?.feed_status ?? "waiting";
  const detectorStatus = snapshot?.detector_status ?? "initializing";
  const modelLoaded = snapshot?.model_loaded ?? false;
  const fftThreshold = snapshot?.fft_threshold ?? settings.fft_threshold;
  const swarmData = {
    ...swarmTelemetry,
    status: snapshot?.status ?? swarmTelemetry.status,
    noise_score: snapshot?.noise_score,
  };

  // Alert system: visual banner + audio beep on attack onset. The onset is
  // detected while rendering (React's "adjust state on prop change" pattern),
  // the effect only handles the beep and the timed dismissal.
  const [prevAttack, setPrevAttack] = useState(attack);
  const [alertActive, setAlertActive] = useState(false);
  if (attack !== prevAttack) {
    setPrevAttack(attack);
    if (attack) setAlertActive(true);
  }
  useEffect(() => {
    if (!alertActive) return undefined;
    playAlertBeep();
    const t = setTimeout(() => setAlertActive(false), 3000);
    return () => clearTimeout(t);
  }, [alertActive]);

  // Load live defense settings once.
  useEffect(() => {
    fetch("/api/settings")
      .then((r) => r.json())
      .then((s) => setSettings((prev) => ({ ...prev, ...s })))
      .catch(() => {});
  }, []);

  const applySettings = useCallback(async (next) => {
    try {
      const headers = { "Content-Type": "application/json" };
      if (API_TOKEN) headers["X-KAAL-Token"] = API_TOKEN;
      const res = await fetch("/api/settings", {
        method: "PUT",
        headers,
        body: JSON.stringify(next),
      });
      if (!res.ok) return false;
      const s = await res.json();
      setSettings((prev) => ({ ...prev, ...s }));
      return true;
    } catch {
      return false;
    }
  }, []);

  return (
    <div className="app flex flex-col h-screen overflow-hidden text-white">
      <Navbar systemMetrics={systemMetrics} isRunning={isRunning} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          systemMetrics={systemMetrics}
        />

        <div className="flex-1 grid grid-cols-8 grid-rows-[auto_2fr_1fr_auto] gap-3 p-3 overflow-hidden">
          <div className="col-span-8">
            <StatusBar
              systemMetrics={systemMetrics}
              isRunning={isRunning}
              swarmData={swarmData}
              connected={connected}
              attack={attack}
              fftThreshold={fftThreshold}
              flaggedCount={flaggedCount}
              temporal={temporal}
              detector={detector}
              operatingMode={operatingMode}
              feedStatus={feedStatus}
              detectorStatus={detectorStatus}
              modelLoaded={modelLoaded}
            />
          </div>

          <div className="col-span-6 row-span-2 glass active">
            <MainPanel
              image={VIDEO_FEED}
              selectedModel={selectedModel}
              operatingMode={operatingMode}
              feedStatus={feedStatus}
              detectorStatus={detectorStatus}
              modelLoaded={modelLoaded}
              swarmData={swarmData}
              onSwarmData={setSwarmTelemetry}
              detections={detections}
              attack={attack}
              alertActive={alertActive}
            />
          </div>

          <div className="col-span-2 glass">
            <GraphPanel data={data} />
          </div>

          <div className="col-span-2 glass">
            <MultiGraph data={data} />
          </div>

          <div className="col-span-4 flex flex-col gap-3 overflow-hidden">
            <div className="flex-1 glass min-h-0">
              <Controls
                setSelectedModel={setSelectedModel}
                isRunning={isRunning}
                setIsRunning={setIsRunning}
                systemMetrics={systemMetrics}
              />
            </div>
            <div className="flex-1 glass min-h-0">
              <SettingsPanel
                key={`${fftThreshold}-${settings.conf}`}
                settings={{ fft_threshold: fftThreshold, conf: settings.conf }}
                onApply={applySettings}
              />
            </div>
          </div>

          <div className="col-span-4">
            <HealthMonitor systemMetrics={systemMetrics} isRunning={isRunning} />
          </div>
        </div>
      </div>
    </div>
  );
}

function playAlertBeep() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.5);
  } catch {
    /* audio unavailable — visual alert still fires */
  }
}

export default App;
