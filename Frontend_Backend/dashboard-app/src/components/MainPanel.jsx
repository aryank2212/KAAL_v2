// MainPanel — centre view: the annotated camera feed (MJPEG /api/video_feed,
// default) or, via the toggle button, the swarm simulation canvas; shows
// detection tags, flagged boxes and a pulsing border while an alert is active.
import { useState } from "react";
import { motion } from "motion/react";
import KaalSwarmCanvas from "./KaalSwarmCanvas";

function MainPanel({
  image,
  selectedModel,
  isRunning,
  swarmData = {},
  onSwarmData,
  detections = [],
  attack = false,
  alertActive = false,
  operatingMode = "INITIALIZING",
  feedStatus = "waiting",
  detectorStatus = "initializing",
  modelLoaded = false,
}) {
  const [isVideo, setIsVideo] = useState(false);
  const flagged = detections.filter((d) => d.is_adversarial);
  const displayMode = isVideo ? "SIMULATION" : operatingMode;
  const modeColor = displayMode === "LIVE" ? "#34d399" : displayMode === "REPLAY" ? "#fbbf24" : "#22d3ee";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="h-full flex flex-col"
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="panel-header mb-0">
            <h3 className="panel-title">MISSION FEED</h3>
          </div>
          <div className="h-3 w-px bg-blue-500/20" />
          <span className="text-[9px] tracking-wider text-slate-600" style={{ fontFamily: "var(--font-mono)" }}>
            {selectedModel.toUpperCase()}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsVideo(!isVideo)}
            className="btn text-[10px] py-1.5 px-3"
          >
            {isVideo ? "◉ IMAGE" : "▶ SIMULATION"}
          </button>
          <div className="panel-dot" />
        </div>
      </div>

      {/* Feed viewport */}
      <div
        className={`relative flex-1 scanlines rounded-xl overflow-hidden border bg-black ${
          attack ? "border-rose-500/60" : "border-blue-500/10"
        } ${alertActive ? "animate-pulse" : ""}`}
      >
        {isVideo ? (
          <div className="w-full h-full relative z-0">
            <KaalSwarmCanvas onTelemetry={onSwarmData} />
          </div>
        ) : (
          <img
            src={image}
            className="w-full h-full object-cover"
            alt="Mission feed"
            style={{ filter: "brightness(0.7) contrast(1.1)" }}
          />
        )}

        {!isVideo && (feedStatus !== "active" || !modelLoaded) && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/55 pointer-events-none">
            <div className="max-w-sm mx-4 rounded-lg border border-amber-400/30 bg-black/85 px-5 py-4 text-center shadow-[0_0_28px_rgba(251,191,36,0.12)]">
              <div className="text-[11px] font-bold tracking-[2px] text-amber-300" style={{ fontFamily: "var(--font-display)" }}>
                {feedStatus !== "active" ? `FEED ${feedStatus.toUpperCase()}` : "DETECTOR UNAVAILABLE"}
              </div>
              <div className="mt-2 text-[10px] leading-relaxed text-slate-300" style={{ fontFamily: "var(--font-mono)" }}>
                {feedStatus !== "active"
                  ? "No current frames are being received. Any previously displayed frame is not current."
                  : "The feed may be visible, but no model inference is available. Detections are not being verified."}
              </div>
              {!modelLoaded && (
                <div className="mt-2 text-[9px] tracking-wider text-rose-300">NO MODEL INFERENCE</div>
              )}
            </div>
          </div>
        )}

        {/* HUD Overlay — Corner brackets */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Top-left bracket */}
          <div className="absolute top-3 left-3">
            <div className="w-6 h-6 border-t-2 border-l-2 border-cyan-400/40 rounded-tl-sm" />
          </div>
          {/* Top-right bracket */}
          <div className="absolute top-3 right-3">
            <div className="w-6 h-6 border-t-2 border-r-2 border-cyan-400/40 rounded-tr-sm" />
          </div>
          {/* Bottom-left bracket */}
          <div className="absolute bottom-3 left-3">
            <div className="w-6 h-6 border-b-2 border-l-2 border-cyan-400/40 rounded-bl-sm" />
          </div>
          {/* Bottom-right bracket */}
          <div className="absolute bottom-3 right-3">
            <div className="w-6 h-6 border-b-2 border-r-2 border-cyan-400/40 rounded-br-sm" />
          </div>

          {/* Center crosshair */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
            <div className="w-8 h-8 relative">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-2 bg-cyan-400/30" />
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-2 bg-cyan-400/30" />
              <div className="absolute left-0 top-1/2 -translate-y-1/2 w-2 h-px bg-cyan-400/30" />
              <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-px bg-cyan-400/30" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full border border-cyan-400/40" />
            </div>
          </div>
        </div>

        {/* HUD Data overlays */}
        {/* LIVE badge */}
        <div className="absolute top-3 left-10 flex items-center gap-2">
          <div
            className="live-dot"
            style={{
              width: 6, height: 6,
              background: feedStatus === "active" ? modeColor : "#fbbf24"
            }}
          />
          <span
            className="text-[10px] tracking-wider font-medium"
            style={{
              fontFamily: "var(--font-display)",
              color: feedStatus === "active" || isVideo ? modeColor : "#fbbf24"
            }}
          >
            {isVideo ? "SIMULATION" : feedStatus === "active" ? displayMode : feedStatus.toUpperCase()}
          </span>
        </div>

        {/* SAT-LINK badge */}
        <div className="absolute top-3 right-10 flex items-center gap-2 px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20">
          <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
          <span className="text-[9px] tracking-wider text-blue-300" style={{ fontFamily: "var(--font-display)" }}>
            {swarmData.mode === 'KAAL' ? 'KAAL MESH ACTIVE' : 'STANDARD OPS'}
          </span>
        </div>

        {/* Bottom data strip */}
        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 py-2 bg-gradient-to-t from-black/80 to-transparent">
          <div className="flex items-center gap-4">
            <DataTag label="SOURCE" value={displayMode} />
            <DataTag label="FEED" value={isVideo ? "SIMULATED" : feedStatus.toUpperCase()} />
          </div>

          <div className="flex items-center gap-4">
            <DataTag label="DETECTOR" value={detectorStatus.toUpperCase()} />
            <DataTag label="INFERENCE" value={modelLoaded ? "READY" : "UNAVAILABLE"} />
          </div>
        </div>

        {/* Top vignette */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/30 via-transparent to-transparent pointer-events-none" />

        {/* ADVERSARIAL ALERT banner */}
        {attack && (
          <motion.div
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-2 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1 rounded bg-rose-600/90 border border-rose-300/50 shadow-[0_0_20px_rgba(244,63,94,0.6)]"
          >
            <div className="w-2 h-2 rounded-full bg-white animate-ping" />
            <span className="text-[10px] tracking-[3px] font-bold text-white" style={{ fontFamily: "var(--font-display)" }}>
              ADVERSARIAL ATTACK DETECTED
            </span>
          </motion.div>
        )}

        {/* Flagged detection readout */}
        {flagged.length > 0 && (
          <div className="absolute top-10 right-3 flex flex-col gap-1 z-10">
            {flagged.map((d, i) => (
              <div
                key={`${d.label}-${i}`}
                className="px-2 py-1 rounded bg-black/70 border border-rose-500/40 text-right"
              >
                <span className="text-[9px] text-rose-300 font-medium" style={{ fontFamily: "var(--font-mono)" }}>
                  {d.label.toUpperCase()}{" "}
                  <span className="text-rose-400">▲ {d.anomaly_score?.toFixed(3)}</span>
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

function DataTag({ label, value }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[8px] tracking-wider text-slate-500" style={{ fontFamily: "var(--font-display)" }}>
        {label}
      </span>
      <span className="text-[10px] font-medium text-cyan-300" style={{ fontFamily: "var(--font-mono)" }}>
        {value}
      </span>
    </div>
  );
}

export default MainPanel;