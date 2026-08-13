import { useState } from "react";
import { motion } from "motion/react";

function Controls({ setSelectedModel, isRunning, setIsRunning }) {
  const [activeBtn, setActiveBtn] = useState(isRunning ? "start" : "stop");

  const handleStart = () => {
    setIsRunning(true);
    setActiveBtn("start");
  };

  const handleStop = () => {
    setIsRunning(false);
    setActiveBtn("stop");
  };

  const handleReset = () => {
    setActiveBtn("reset");
    setTimeout(() => setActiveBtn(isRunning ? "start" : "stop"), 500);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="h-full flex flex-col"
    >
      {/* Header */}
      <div className="panel-header">
        <h3 className="panel-title">CONTROL SYSTEM</h3>
        <div className="panel-dot" />
      </div>

      {/* Model selector */}
      <div className="relative mb-3">
        <label className="text-[8px] tracking-[2px] text-slate-600 block mb-1.5" style={{ fontFamily: "var(--font-display)" }}>
          TARGET MODEL
        </label>
        <div className="relative">
          <select
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full p-2.5 bg-[#060a12] border border-blue-500/15 rounded-lg text-[11px] tracking-wider focus:outline-none focus:border-blue-500/40 transition-colors"
            style={{ fontFamily: "var(--font-mono)" }}
          >
            <option value="Model 1">QUANTUM-V1 — Neural Mesh</option>
            <option value="Model 2">HELIX-V2 — Deep Resonance</option>
            <option value="Model 3">ATLAS-V3 — Spatial Map</option>
          </select>
          {/* Custom arrow */}
          <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
            <svg className="w-3 h-3 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {/* Control buttons */}
      <div className="flex gap-2 mb-4">
        <ControlButton
          icon="▶"
          label="START"
          color="emerald"
          isActive={activeBtn === "start"}
          onClick={handleStart}
        />
        <ControlButton
          icon="■"
          label="STOP"
          color="rose"
          isActive={activeBtn === "stop"}
          onClick={handleStop}
        />
        <ControlButton
          icon="↺"
          label="RESET"
          color="amber"
          isActive={activeBtn === "reset"}
          onClick={handleReset}
        />
      </div>

      {/* Metrics row */}
      <div className="flex-1" />
      <div className="flex items-end justify-between">
        <div>
          <p className="metric text-xl">{Math.floor(Math.random() * 100)}%</p>
          <p className="metric-label">CPU LOAD</p>
        </div>
        <div className="text-right">
          <p className="text-sm font-medium text-emerald-400" style={{ fontFamily: "var(--font-mono)" }}>
            {isRunning ? "RUNNING" : "STOPPED"}
          </p>
          <p className="text-[8px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>
            SIM STATE
          </p>
        </div>
      </div>
    </motion.div>
  );
}

function ControlButton({ icon, label, color, isActive, onClick }) {
  const colorMap = {
    emerald: {
      bg: isActive ? "bg-emerald-500/10" : "bg-transparent",
      border: isActive ? "border-emerald-500/40" : "border-blue-500/15",
      text: isActive ? "text-emerald-300" : "text-slate-500",
      glow: isActive ? "shadow-[0_0_15px_rgba(52,211,153,0.15)]" : "",
    },
    rose: {
      bg: isActive ? "bg-rose-500/10" : "bg-transparent",
      border: isActive ? "border-rose-500/40" : "border-blue-500/15",
      text: isActive ? "text-rose-300" : "text-slate-500",
      glow: isActive ? "shadow-[0_0_15px_rgba(244,63,94,0.15)]" : "",
    },
    amber: {
      bg: isActive ? "bg-amber-500/10" : "bg-transparent",
      border: isActive ? "border-amber-500/40" : "border-blue-500/15",
      text: isActive ? "text-amber-300" : "text-slate-500",
      glow: isActive ? "shadow-[0_0_15px_rgba(251,191,36,0.15)]" : "",
    },
  };

  const c = colorMap[color];

  return (
    <motion.button
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded-lg border transition-all duration-300 ${c.bg} ${c.border} ${c.text} ${c.glow} hover:brightness-125`}
    >
      <span className="text-xs">{icon}</span>
      <span className="text-[10px] tracking-[2px] font-medium" style={{ fontFamily: "var(--font-display)" }}>
        {label}
      </span>
    </motion.button>
  );
}

export default Controls;