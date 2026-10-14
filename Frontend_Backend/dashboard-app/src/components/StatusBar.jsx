// StatusBar — live readouts: flagged boxes, frame FFT score vs threshold,
// temporal cue (OFF / CALIB / residual), DETECTOR backend, connection state
// and CPU / GPU / memory / latency / uptime from systemMetrics.
import { motion } from "motion/react";

export default function StatusBar({
  systemMetrics,
  isRunning,
  swarmData = {},
  connected = false,
  attack = false,
  fftThreshold = null,
  flaggedCount = 0,
  temporal = null,
  detector = null,
  operatingMode = "INITIALIZING",
  feedStatus = "waiting",
  detectorStatus = "initializing",
  modelLoaded = false,
}) {
  const noise = swarmData.noise_score || 0;
  const noiseHot = fftThreshold !== null && noise > fftThreshold;
  const temporalValue = !temporal ? "OFF" : temporal.ready ? temporal.score.toFixed(4) : "CALIB";
  const metrics = [
    { label: "FLAGS", value: flaggedCount, color: flaggedCount > 0 ? "#f43f5e" : "#34d399", icon: "⬡" },
    { label: "NOISE", value: typeof noise === "number" ? noise.toFixed(3) : 0, color: noiseHot ? "#f43f5e" : "#fbbf24", icon: "⚡" },
    { label: "THRESH", value: fftThreshold !== null ? fftThreshold.toFixed(3) : "—", color: "#22d3ee", icon: "⊘" },
    { label: "TEMPORAL", value: temporalValue, color: temporal?.flagged ? "#f43f5e" : "#a78bfa", icon: "◷" },
    { label: "DETECTOR", value: !modelLoaded ? detectorStatus.toUpperCase() : (detector ? detector.backend.toUpperCase() : "READY"), color: !modelLoaded ? "#fbbf24" : "#cbd5e1", icon: "◉" },
    { label: "MODE", value: operatingMode, color: operatingMode === "LIVE" ? "#34d399" : operatingMode === "REPLAY" ? "#fbbf24" : operatingMode === "SIMULATION" ? "#22d3ee" : "#cbd5e1", icon: "⏣" },
    { label: "CPU", value: `${Math.round(systemMetrics.cpu)}%`, color: systemMetrics.cpu > 80 ? "#f43f5e" : "#22d3ee", icon: "⬡" },
    { label: "MEM", value: `${Math.round(systemMetrics.memory)}%`, color: systemMetrics.memory > 85 ? "#f43f5e" : "#3b82f6", icon: "◈" },
    { label: "GPU", value: systemMetrics.gpu !== null && systemMetrics.gpu !== undefined ? `${Math.round(systemMetrics.gpu)}%` : "N/A", color: systemMetrics.gpu > 80 ? "#fbbf24" : "#34d399", icon: "◇" },
    { label: "LATENCY", value: `${Math.round(systemMetrics.latency)}ms`, color: systemMetrics.latency > 100 ? "#fbbf24" : "#22d3ee", icon: "◎" },
    { label: "UPTIME", value: systemMetrics.uptime, color: "#34d399", icon: "⏣" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-red-500/8 bg-[#0a0406]/60"
    >
      {/* Live indicator */}
      <div className="flex items-center gap-2 pr-3 border-r border-red-500/10">
        <div
          className="w-1.5 h-4 rounded-full"
          style={{
            background: attack ? "#f43f5e" : !isRunning || feedStatus !== "active" ? "#fbbf24" : "#34d399",
            boxShadow: `0 0 8px ${attack ? "rgba(244,63,94,0.7)" : !isRunning || feedStatus !== "active" ? "rgba(251,191,36,0.5)" : "rgba(52,211,153,0.5)"}`,
          }}
        />
        <span className="text-[9px] tracking-[3px] text-slate-500" style={{ fontFamily: "var(--font-display)" }}>
          {attack ? "ALERT" : !isRunning ? "PAUSED" : `FEED ${feedStatus.toUpperCase()}`}
        </span>
        {connected && (
          <span className="text-[8px] tracking-[2px] text-cyan-400/70" style={{ fontFamily: "var(--font-mono)" }}>
            WS
          </span>
        )}
      </div>

      {metrics.map((m, i) => (
        <div key={m.label} className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] opacity-40">{m.icon}</span>
            <span className="text-[8px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>
              {m.label}
            </span>
            <span
              className="text-[11px] font-medium"
              style={{ fontFamily: "var(--font-mono)", color: m.color, textShadow: `0 0 6px ${m.color}40` }}
            >
              {m.value}
            </span>
          </div>
          {i < metrics.length - 1 && <div className="w-px h-3 bg-red-500/8" />}
        </div>
      ))}

      <div className="flex-1" />

      <div className="flex items-center gap-1.5">
        <div className="w-1 h-1 rounded-full bg-red-400/50" />
        <span className="text-[8px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>
          KAAL v2.4.1
        </span>
      </div>
    </motion.div>
  );
}
