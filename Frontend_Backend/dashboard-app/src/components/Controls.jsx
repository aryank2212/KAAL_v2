// Controls — pauses/resumes this browser's dashboard stream only.
// The backend camera worker continues running while the view is paused.
import { useState, useEffect } from "react";
import { motion } from "motion/react";

function Controls({ isRunning, setIsRunning, systemMetrics }) {
  const [activeBtn, setActiveBtn] = useState(isRunning ? "resume" : "pause");

  useEffect(() => {
    setActiveBtn(isRunning ? "resume" : "pause");
  }, [isRunning]);

  const handleResume = () => {
    setIsRunning(true);
    setActiveBtn("resume");
  };

  const handlePause = () => {
    setIsRunning(false);
    setActiveBtn("pause");
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="h-full flex flex-col"
    >
      <div className="panel-header">
        <h3 className="panel-title">DASHBOARD STREAM</h3>
        <div className="panel-dot" />
      </div>

      <p className="text-[9px] leading-relaxed text-slate-500 mb-3">
        These controls pause or resume updates in this browser. They do not start or stop the backend camera or inference worker.
      </p>

      <div className="flex gap-2 mb-4">
        <ControlButton
          icon="▶"
          label="RESUME VIEW"
          color="emerald"
          isActive={activeBtn === "resume"}
          onClick={handleResume}
        />
        <ControlButton
          icon="Ⅱ"
          label="PAUSE VIEW"
          color="rose"
          isActive={activeBtn === "pause"}
          onClick={handlePause}
        />
      </div>

      <div className="flex-1" />
      <div className="flex items-end justify-between">
        <div>
          <p className="metric text-xl">{Math.round(systemMetrics?.cpu ?? 0)}%</p>
          <p className="metric-label">HOST CPU</p>
        </div>
        <div className="text-right">
          <p className={`text-sm font-medium ${isRunning ? "text-emerald-400" : "text-amber-300"}`}
             style={{ fontFamily: "var(--font-mono)" }}>
            {isRunning ? "VIEW ACTIVE" : "VIEW PAUSED"}
          </p>
          <p className="text-[8px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>
            CLIENT STATE
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
