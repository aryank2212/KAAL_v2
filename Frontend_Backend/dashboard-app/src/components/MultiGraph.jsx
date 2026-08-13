import { motion } from "motion/react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";

function MultiGraph({ data }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.35 }}
      className="h-full flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <div>
          <div className="panel-header mb-0">
            <h3 className="panel-title">PERFORMANCE</h3>
          </div>
          <p className="text-[9px] tracking-wider text-slate-600" style={{ fontFamily: "var(--font-mono)" }}>
            MULTI-METRIC ANALYSIS
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Mini legend */}
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-0.5 rounded bg-blue-500" />
            <span className="text-[8px] text-slate-500" style={{ fontFamily: "var(--font-mono)" }}>α</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-0.5 rounded bg-emerald-400" />
            <span className="text-[8px] text-slate-500" style={{ fontFamily: "var(--font-mono)" }}>β</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-0.5 rounded bg-violet-400" />
            <span className="text-[8px] text-slate-500" style={{ fontFamily: "var(--font-mono)" }}>γ</span>
          </div>
          <div className="panel-dot" />
        </div>
      </div>

      {/* Graph */}
      <div className="flex-1 -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="alphaGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="betaGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#34d399" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gammaGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(59,130,246,0.06)"
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
              cursor={{ stroke: "rgba(59,130,246,0.15)" }}
            />

            <Area
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={1.5}
              fill="url(#alphaGradient)"
              dot={false}
              name="Signal α"
            />
            <Area
              type="monotone"
              dataKey="value2"
              stroke="#34d399"
              strokeWidth={1.5}
              fill="url(#betaGradient)"
              dot={false}
              name="Signal β"
            />
            <Area
              type="monotone"
              dataKey="value3"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              fill="url(#gammaGradient)"
              dot={false}
              name="Signal γ"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

export default MultiGraph;