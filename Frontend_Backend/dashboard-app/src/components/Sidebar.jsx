// Sidebar — actual detector/source health; no cosmetic model selectors.
import { motion } from "motion/react";

export default function Sidebar({
  systemMetrics,
  detector = null,
  detectorHealth = "INITIALIZING",
  feedState = "INITIALIZING",
}) {
  const detectorReady = detectorHealth === "READY";
  const detectorDemo = detectorHealth === "DEMO";
  const healthColor = detectorReady ? "bg-emerald-400" : detectorDemo ? "bg-amber-400" : "bg-rose-400";
  const healthText = detectorReady ? "READY" : detectorDemo ? "DEMO FALLBACK" : detectorHealth;

  const navItems = [
    { icon: "◈", label: "OVERVIEW", active: true },
    { icon: "◇", label: "ANALYTICS", active: false },
    { icon: "⬡", label: "MODELS", active: false },
    { icon: "◉", label: "SETTINGS", active: false },
  ];

  return (
    <motion.div
      initial={{ x: -60, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="w-56 min-w-[220px] h-full flex flex-col border-r border-blue-500/10 bg-gradient-to-b from-[#060a12] to-[#030508]"
      style={{ padding: "16px 12px" }}
    >
      <div className="mb-6">
        <p className="text-[8px] tracking-[3px] text-slate-600 mb-3 px-2" style={{ fontFamily: "var(--font-display)" }}>
          NAVIGATION
        </p>
        <div className="flex flex-col gap-0.5">
          {navItems.map((item) => (
            <div
              key={item.label}
              aria-current={item.active ? "page" : undefined}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-left border ${
                item.active
                  ? "bg-blue-500/10 text-blue-300 border-blue-500/20"
                  : "text-slate-500 border-transparent"
              }`}
            >
              <span className="text-xs opacity-60">{item.icon}</span>
              <span className="text-[10px] tracking-[2px] font-medium" style={{ fontFamily: "var(--font-display)" }}>
                {item.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="h-px bg-gradient-to-r from-transparent via-blue-500/15 to-transparent mb-5" />

      <div className="mb-6">
        <p className="text-[8px] tracking-[3px] text-slate-600 mb-3 px-2" style={{ fontFamily: "var(--font-display)" }}>
          DETECTOR STATUS
        </p>
        <div className="px-3 py-3 rounded-lg border border-blue-500/15 bg-blue-500/[0.03]">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[11px] tracking-wider font-medium text-white" style={{ fontFamily: "var(--font-display)" }}>
              {detector?.backend ? detector.backend.toUpperCase() : "DETECTOR"}
            </p>
            <span className={`w-1.5 h-1.5 rounded-full ${healthColor}`} />
          </div>
          <p className="text-[9px] text-slate-500 mt-1 break-all" style={{ fontFamily: "var(--font-mono)" }}>
            {detector?.weights || "weights unavailable"}
          </p>
          <p className={`text-[9px] mt-2 ${detectorReady ? "text-emerald-400" : detectorDemo ? "text-amber-300" : "text-rose-300"}`}
             style={{ fontFamily: "var(--font-display)" }}>
            {healthText}
          </p>
          {detector?.fallback_reason && (
            <p className="text-[8px] text-amber-300/80 mt-1">{detector.fallback_reason}</p>
          )}
        </div>
      </div>

      <div className="flex-1" />

      <div className="space-y-3">
        <div className="h-px bg-gradient-to-r from-transparent via-blue-500/15 to-transparent" />
        <div className="grid grid-cols-2 gap-2">
          <MiniMetric label="CPU" value={`${Math.round(systemMetrics?.cpu || 0)}%`} />
          <MiniMetric label="MEM" value={`${Math.round(systemMetrics?.memory || 0)}%`} />
        </div>
        <div className="px-2 py-2 rounded-lg bg-white/[0.02] border border-blue-500/8">
          <p className="text-[7px] tracking-[2px] text-slate-600 mb-1" style={{ fontFamily: "var(--font-display)" }}>
            FEED STATE
          </p>
          <p className="text-[10px] tracking-wider text-cyan-300 break-words" style={{ fontFamily: "var(--font-mono)" }}>
            {feedState}
          </p>
          <p className="text-[8px] text-slate-600 mt-1">KAAL V3 hardening track</p>
        </div>
      </div>
    </motion.div>
  );
}

function MiniMetric({ label, value }) {
  return (
    <div className="px-2 py-1.5 rounded-md bg-white/[0.02] border border-blue-500/8">
      <p className="text-[7px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>{label}</p>
      <p className="text-[12px] font-medium text-cyan-400" style={{ fontFamily: "var(--font-mono)" }}>{value}</p>
    </div>
  );
}
