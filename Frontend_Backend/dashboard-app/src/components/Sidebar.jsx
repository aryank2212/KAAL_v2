// Sidebar — navigation, a model list and mini metrics. The model entries
// (QUANTUM-V1 / HELIX-V2 / ATLAS-V3) are cosmetic labels shown in the main
// panel header; the real detector is chosen on the backend (KAAL_DETECTOR /
// KAAL_MODEL_PATH) and shown in the StatusBar DETECTOR item.
import { motion } from "motion/react";

export default function Sidebar({ selectedModel, setSelectedModel, systemMetrics }) {
  const models = [
    { id: "Model 1", label: "QUANTUM-V1", desc: "Neural Mesh", status: "ready" },
    { id: "Model 2", label: "HELIX-V2", desc: "Deep Resonance", status: "ready" },
    { id: "Model 3", label: "ATLAS-V3", desc: "Spatial Map", status: "training" },
  ];

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
      {/* Nav Section */}
      <div className="mb-6">
        <p className="text-[8px] tracking-[3px] text-slate-600 mb-3 px-2" style={{ fontFamily: "var(--font-display)" }}>
          NAVIGATION
        </p>
        <div className="flex flex-col gap-0.5">
          {navItems.map((item) => (
            <button
              key={item.label}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all duration-200 ${
                item.active
                  ? "bg-blue-500/10 text-blue-300 border border-blue-500/20"
                  : "text-slate-500 hover:text-slate-300 hover:bg-white/[0.02] border border-transparent"
              }`}
            >
              <span className="text-xs opacity-60">{item.icon}</span>
              <span className="text-[10px] tracking-[2px] font-medium" style={{ fontFamily: "var(--font-display)" }}>
                {item.label}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-gradient-to-r from-transparent via-blue-500/15 to-transparent mb-5" />

      {/* Models Section */}
      <div className="mb-6">
        <p className="text-[8px] tracking-[3px] text-slate-600 mb-3 px-2" style={{ fontFamily: "var(--font-display)" }}>
          ACTIVE MODELS
        </p>
        <div className="flex flex-col gap-1.5">
          {models.map((model) => {
            const isSelected = selectedModel === model.id;
            return (
              <motion.button
                key={model.id}
                whileTap={{ scale: 0.98 }}
                onClick={() => setSelectedModel(model.id)}
                className={`relative w-full text-left px-3 py-2.5 rounded-lg transition-all duration-300 border ${
                  isSelected
                    ? "bg-gradient-to-r from-blue-500/10 to-cyan-500/5 border-blue-500/30 shadow-[0_0_20px_rgba(59,130,246,0.08)]"
                    : "bg-transparent border-transparent hover:bg-white/[0.02] hover:border-blue-500/10"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p
                      className={`text-[11px] tracking-wider font-medium ${isSelected ? "text-white" : "text-slate-400"}`}
                      style={{ fontFamily: "var(--font-display)" }}
                    >
                      {model.label}
                    </p>
                    <p className="text-[9px] text-slate-600 mt-0.5" style={{ fontFamily: "var(--font-mono)" }}>
                      {model.desc}
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div
                      className={`w-1.5 h-1.5 rounded-full ${
                        model.status === "ready"
                          ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]"
                          : "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.5)] animate-pulse"
                      }`}
                    />
                  </div>
                </div>
                {/* Active indicator bar */}
                {isSelected && (
                  <motion.div
                    layoutId="sidebar-active"
                    className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-blue-400 shadow-[0_0_8px_rgba(59,130,246,0.6)]"
                  />
                )}
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Bottom System Info */}
      <div className="space-y-3">
        <div className="h-px bg-gradient-to-r from-transparent via-blue-500/15 to-transparent" />

        {/* Mini metrics */}
        <div className="grid grid-cols-2 gap-2">
          <MiniMetric label="CPU" value={`${Math.round(systemMetrics?.cpu || 0)}%`} />
          <MiniMetric label="MEM" value={`${Math.round(systemMetrics?.memory || 0)}%`} />
        </div>

        {/* Status row */}
        <div className="flex items-center justify-between px-2 py-2 rounded-lg bg-white/[0.02] border border-blue-500/8">
          <div className="flex items-center gap-2">
            <div className="live-dot" style={{ width: 6, height: 6 }} />
            <span className="text-[9px] tracking-[2px] text-slate-500" style={{ fontFamily: "var(--font-display)" }}>
              CONNECTED
            </span>
          </div>
          <span className="text-[9px] text-slate-600" style={{ fontFamily: "var(--font-mono)" }}>
            v2.4
          </span>
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