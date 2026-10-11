// GraphPanel — area chart of the frame-level FFT anomaly score over the last
// KAAL_HISTORY snapshots (data[].value against data[].time).
import { motion } from "motion/react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

function GraphPanel({ data }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="h-full flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <div>
          <div className="panel-header mb-0">
            <h3 className="panel-title">ANALYTICS</h3>
          </div>
          <p className="text-[9px] tracking-wider text-slate-600" style={{ fontFamily: "var(--font-mono)" }}>
            REAL-TIME SIGNAL
          </p>
        </div>
        <div className="panel-dot" />
      </div>

      {/* Current value display */}
      <div className="flex items-baseline gap-2 mb-2">
        <span className="text-lg font-semibold text-cyan-300" style={{ fontFamily: "var(--font-mono)" }}>
          {data.length > 0 ? data[data.length - 1].value.toFixed(1) : "—"}
        </span>
        <span className="text-[9px] tracking-wider text-slate-500" style={{ fontFamily: "var(--font-display)" }}>
          UNITS
        </span>
      </div>

      {/* Graph */}
      <div className="flex-1 -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="signalGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="50%" stopColor="#3b82f6" stopOpacity={0.1} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(59, 130, 246, 0.06)"
              vertical={false}
            />
            <XAxis
              dataKey="time"
              stroke="transparent"
              tick={{ fontSize: 9, fill: "#334155", fontFamily: "var(--font-mono)" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              stroke="transparent"
              tick={{ fontSize: 9, fill: "#334155", fontFamily: "var(--font-mono)" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0a0f1a",
                border: "1px solid rgba(59,130,246,0.2)",
                borderRadius: "8px",
                fontSize: "11px",
                fontFamily: "var(--font-mono)",
                color: "#e2e8f0",
                boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
              }}
              cursor={{ stroke: "rgba(59,130,246,0.2)" }}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="url(#signalGradient)"
              dot={false}
              activeDot={{ r: 3, stroke: "#3b82f6", strokeWidth: 2, fill: "#0a0f1a" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

export default GraphPanel;