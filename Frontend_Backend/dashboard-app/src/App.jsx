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

function App() {
  const [selectedModel, setSelectedModel] = useState("Model 1");
  const [data, setData] = useState([]);
  const [image, setImage] = useState(null);
  const [isRunning, setIsRunning] = useState(true);
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 0,
    memory: 0,
    gpu: null,
    bandwidth: 0,
    latency: 0,
    uptime: "00:00:00",
  });
  const [detections, setDetections] = useState([]);
  const [attack, setAttack] = useState(false);
  const [alertActive, setAlertActive] = useState(false);
  const [settings, setSettings] = useState({ fft_threshold: 0.5, conf: 0.4 });
  const [swarmData, setSwarmData] = useState({
    status: "NOMINAL",
    alive: 0,
    kia: 0,
    mode: "NORMAL",
  });

  // Real-time stream (WebSocket with polling fallback).
  const { snapshot, connected } = useDefenseSocket(isRunning ? 1000 : null);

  useEffect(() => {
    setImage("/api/video_feed");
  }, [selectedModel]);

  // Apply incoming snapshot to dashboard state.
  useEffect(() => {
    if (!snapshot) return;
    setData(snapshot.data || []);
    setSystemMetrics(snapshot.systemMetrics || systemMetrics);
    setDetections(snapshot.detections || []);
    setAttack(Boolean(snapshot.is_attack));
    setSettings((prev) => ({
      ...prev,
      fft_threshold: snapshot.fft_threshold ?? prev.fft_threshold,
    }));
    setSwarmData((prev) => ({
      ...prev,
      status: snapshot.status || prev.status,
      noise_score: snapshot.noise_score,
      alive: snapshot.flagged_count !== undefined ? snapshot.flagged_count : prev.alive,
      kia: snapshot.flagged_count !== undefined ? snapshot.flagged_count : prev.kia,
    }));
  }, [snapshot]);

  // Alert system: visual banner + audio beep on attack onset.
  useEffect(() => {
    if (!attack) return;
    setAlertActive(true);
    playAlertBeep();
    const t = setTimeout(() => setAlertActive(false), 3000);
    return () => clearTimeout(t);
  }, [attack]);

  // Load live defense settings once.
  useEffect(() => {
    fetch("/api/settings")
      .then((r) => r.json())
      .then((s) => setSettings((prev) => ({ ...prev, ...s })))
      .catch(() => {});
  }, []);

  const applySettings = useCallback(async (next) => {
    try {
      const res = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
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
              fftThreshold={settings.fft_threshold}
              flaggedCount={swarmData.kia}
            />
          </div>

          <div className="col-span-6 row-span-2 glass active">
            <MainPanel
              image={image}
              selectedModel={selectedModel}
              isRunning={isRunning}
              swarmData={swarmData}
              onSwarmData={setSwarmData}
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
              />
            </div>
            <div className="flex-1 glass min-h-0">
              <SettingsPanel settings={settings} onApply={applySettings} />
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
