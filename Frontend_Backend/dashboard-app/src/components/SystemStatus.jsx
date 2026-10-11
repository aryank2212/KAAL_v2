// SystemStatus (exported as HealthMonitor) — health bars for CPU, GPU,
// memory, latency and bandwidth from the backend's 1 Hz MetricsSampler.
import { motion } from "motion/react";

// Decorative star field, generated once at module load (not during render).
const STARS = Array.from({ length: 12 }, (_, id) => ({
  id,
  top: Math.random() * 100,
  left: Math.random() * 100,
  size: Math.random() * 2 + 0.5,
  delay: Math.random() * 3,
  duration: 1.5 + Math.random() * 2,
}));

export default function HealthMonitor({ systemMetrics }) {
  const stars = STARS;

  const gauges = [
    { label: "CPU", value: Math.round(systemMetrics.cpu), max: 100, unit: "%", color: "#dc2626" },
    { label: "MEM", value: Math.round(systemMetrics.memory), max: 100, unit: "%", color: "#ef4444" },
    { label: "GPU", value: Math.round(systemMetrics.gpu), max: 100, unit: "%", color: "#34d399" },
    { label: "NET", value: Math.round(systemMetrics.bandwidth), max: 1024, unit: "KB/s", color: "#8b5cf6" },
    { label: "LAT", value: Math.round(systemMetrics.latency), max: 200, unit: "ms", color: "#22d3ee" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: 0.5 }}
      className="relative px-3 py-2.5 rounded-lg border border-red-500/10 bg-[#0a0406]/70 overflow-hidden"
    >
      {/* Twinkling stars */}
      {stars.map((star) => (
        <motion.div
          key={star.id}
          className="absolute rounded-full bg-white pointer-events-none"
          style={{
            top: `${star.top}%`,
            left: `${star.left}%`,
            width: star.size,
            height: star.size,
          }}
          animate={{ opacity: [0.1, 0.8, 0.1] }}
          transition={{
            duration: star.duration,
            repeat: Infinity,
            delay: star.delay,
            ease: "easeInOut",
          }}
        />
      ))}

      {/* Header */}
      <div className="flex items-center gap-2 mb-2 relative z-10">
        <span className="text-[8px] text-red-400/60">✦</span>
        <h4
          className="text-[8px] tracking-[3px] text-slate-500 font-medium"
          style={{ fontFamily: "var(--font-display)" }}
        >
          HEALTH MONITORING SYSTEM
        </h4>
        <span className="text-[8px] text-red-400/60">✦</span>
        <div className="flex-1" />
        <div className="w-1 h-1 rounded-full bg-red-400/50 animate-pulse" />
      </div>

      {/* Compact gauges row */}
      <div className="flex items-center gap-3 relative z-10">
        {gauges.map((g) => {
          const pct = Math.min((g.value / g.max) * 100, 100);
          return (
            <div key={g.label} className="flex items-center gap-2 flex-1">
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-[7px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>
                    {g.label}
                  </span>
                  <span
                    className="text-[10px] font-medium"
                    style={{ fontFamily: "var(--font-mono)", color: g.color, textShadow: `0 0 6px ${g.color}30` }}
                  >
                    {g.value}<span className="text-[8px] text-slate-600">{g.unit}</span>
                  </span>
                </div>
                <div className="h-0.5 rounded-full bg-white/[0.04] overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="h-full rounded-full"
                    style={{
                      background: `linear-gradient(90deg, ${g.color}50, ${g.color})`,
                      boxShadow: `0 0 4px ${g.color}30`,
                    }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
