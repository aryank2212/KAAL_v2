import { useState } from "react";
import { motion } from "motion/react";

// SettingsPanel — Phase 4: live-adjustable defense parameters.
// Slides FFT threshold + detector confidence and pushes them to the backend
// via PUT /api/settings so changes apply without restarting the pipeline.

export default function SettingsPanel({ settings = {}, onApply }) {
  const [fft, setFft] = useState(settings.fft_threshold ?? 0.08);
  const [conf, setConf] = useState(settings.conf ?? 0.4);
  const [saved, setSaved] = useState(false);

  const handleChange = (key, value) => {
    if (key === "fft_threshold") setFft(value);
    else setConf(value);
    setSaved(false);
  };

  const apply = async () => {
    if (onApply) {
      const ok = await onApply({ fft_threshold: fft, conf });
      setSaved(ok);
      if (ok) setTimeout(() => setSaved(false), 1500);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.35 }}
      className="h-full flex flex-col"
    >
      <div className="panel-header">
        <h3 className="panel-title">DEFENSE SETTINGS</h3>
        <div className="panel-dot" />
      </div>

      <Slider
        label="FFT THRESHOLD"
        value={fft}
        min={0.0}
        max={0.5}
        step={0.005}
        display={fft.toFixed(3)}
        accent="#22d3ee"
        onChange={(v) => handleChange("fft_threshold", v)}
      />

      <Slider
        label="DETECTOR CONFIDENCE"
        value={conf}
        min={0.05}
        max={0.95}
        step={0.05}
        display={conf.toFixed(2)}
        accent="#34d399"
        onChange={(v) => handleChange("conf", v)}
      />

      <div className="flex-1" />

      <motion.button
        whileTap={{ scale: 0.97 }}
        onClick={apply}
        className="w-full py-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 text-[10px] tracking-[2px] font-medium hover:brightness-125 transition-all"
        style={{ fontFamily: "var(--font-display)" }}
      >
        {saved ? "✓ SAVED" : "APPLY SETTINGS"}
      </motion.button>
    </motion.div>
  );
}

function Slider({ label, value, min, max, step, display, accent, onChange }) {
  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[8px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>
          {label}
        </span>
        <span className="text-[10px] font-medium" style={{ fontFamily: "var(--font-mono)", color: accent }}>
          {display}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none bg-white/[0.06]"
        style={{ accentColor: accent }}
      />
    </div>
  );
}
