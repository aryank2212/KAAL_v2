// Navbar — top bar: clock / date, pipeline state and network throughput
// (systemMetrics.bandwidth, KB/s).
import { useState, useEffect } from "react";
import { motion } from "motion/react";

export default function Navbar({ systemMetrics, isRunning, operatingMode = "INITIALIZING", feedStatus = "waiting", modelLoaded = false }) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = (d) =>
    d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const formatDate = (d) =>
    d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "2-digit" }).toUpperCase();

  return (
    <motion.nav
      initial={{ y: -40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="flex items-center justify-between px-5 py-2 border-b border-red-500/10 bg-gradient-to-r from-[#0a0406] via-[#0f0810] to-[#0a0406] relative z-10"
      style={{ minHeight: 48 }}
    >
      {/* Left — Brand */}
      <div className="flex items-center gap-3">
        {/* Logo hex — reddish-black */}
        <div className="relative w-8 h-8 flex items-center justify-center">
          <svg viewBox="0 0 32 32" className="w-8 h-8 absolute">
            <polygon
              points="16,1 30,8.5 30,23.5 16,31 2,23.5 2,8.5"
              fill="none"
              stroke="rgba(220,38,38,0.6)"
              strokeWidth="1.5"
            />
            <polygon
              points="16,6 25,11 25,21 16,26 7,21 7,11"
              fill="rgba(220,38,38,0.1)"
              stroke="rgba(220,38,38,0.35)"
              strokeWidth="1"
            />
          </svg>
          <span className="relative text-red-400 text-xs font-bold" style={{ fontFamily: "var(--font-display)" }}>K</span>
        </div>
        <div>
          <h1
            className="text-sm font-bold tracking-[4px] text-white/90"
            style={{ fontFamily: "var(--font-display)" }}
          >
            KAAL
          </h1>
          <p className="text-[8px] tracking-[3px] text-slate-500 -mt-0.5">SIMULATION COMMAND</p>
        </div>
      </div>

      {/* Center — Status Indicators */}
      <div className="flex items-center gap-6">
        <NavIndicator label="SOURCE" value={operatingMode} color={operatingMode === "LIVE" ? "emerald" : operatingMode === "REPLAY" ? "amber" : "cyan"} />
        <NavIndicator label="FEED" value={feedStatus.toUpperCase()} color={feedStatus === "active" ? "emerald" : "amber"} />
        <NavIndicator label="MODEL" value={modelLoaded ? "READY" : "UNAVAILABLE"} color={modelLoaded ? "emerald" : "amber"} />
        <NavIndicator label="NET" value={`${Math.round(systemMetrics.bandwidth)} KB/s`} color="cyan" />

        {/* Divider */}
        <div className="w-px h-5 bg-red-500/15" />

        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]' : 'bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.6)]'}`} />
          <span className="text-[10px] tracking-widest text-slate-400" style={{ fontFamily: "var(--font-display)" }}>
            {isRunning ? "UPDATES ON" : "PAUSED"}
          </span>
        </div>
      </div>

      {/* Right — Clock & User */}
      <div className="flex items-center gap-5">
        {/* Notification bell */}
        <div className="relative cursor-pointer group">
          <svg className="w-4 h-4 text-slate-500 group-hover:text-red-400 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 005.454-1.31A8.967 8.967 0 0118 9.75v-.7V9A6 6 0 006 9v.75a8.967 8.967 0 01-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 01-5.714 0m5.714 0a3 3 0 11-5.714 0" />
          </svg>
          <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-rose-500 rounded-full shadow-[0_0_6px_rgba(244,63,94,0.6)] animate-pulse" />
        </div>

        {/* Clock */}
        <div className="text-right">
          <p className="text-sm font-medium tracking-wider text-white/80" style={{ fontFamily: "var(--font-mono)" }}>
            {formatTime(time)}
          </p>
          <p className="text-[8px] tracking-[2px] text-slate-500">{formatDate(time)}</p>
        </div>

        {/* User avatar — reddish-black */}
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-red-500/20 to-rose-900/20 border border-red-500/20 flex items-center justify-center cursor-pointer hover:border-red-500/40 transition-colors">
          <span className="text-[10px] font-semibold text-red-300" style={{ fontFamily: "var(--font-display)" }}>OP</span>
        </div>
      </div>

      {/* Bottom glow line — reddish */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-red-500/30 to-transparent" />
    </motion.nav>
  );
}

function NavIndicator({ label, value, color }) {
  const colorMap = {
    emerald: "text-emerald-400",
    cyan: "text-cyan-400",
    red: "text-red-400",
    amber: "text-amber-400",
  };

  return (
    <div className="flex items-center gap-2">
      <span className="text-[9px] tracking-widest text-slate-600" style={{ fontFamily: "var(--font-display)" }}>{label}</span>
      <span className={`text-[10px] tracking-wider font-medium ${colorMap[color]}`} style={{ fontFamily: "var(--font-mono)" }}>
        {value}
      </span>
    </div>
  );
}
