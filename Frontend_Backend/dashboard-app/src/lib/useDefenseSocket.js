// useDefenseSocket — real-time data source for the dashboard.
//
// Connects to the backend `/ws` WebSocket for snapshot broadcasts. While the
// socket is down (backend restarting, network blip) it polls
// `/api/python-data` so the dashboard keeps updating, and it keeps retrying the
// socket with exponential backoff; polling stops as soon as the socket is back.

import { useEffect, useRef, useState } from "react";

const MAX_BACKOFF_MS = 10000;

export default function useDefenseSocket(intervalMs = 1000) {
  const [snapshot, setSnapshot] = useState(null);
  const [connected, setConnected] = useState(false);
  const pollRef = useRef(null);
  const enabled = intervalMs !== null && intervalMs !== false;

  useEffect(() => {
    if (!enabled) return undefined;
    let disposed = false;
    let ws = null;
    let retryTimer = null;
    let backoff = 1000;

    const startPolling = () => {
      if (pollRef.current) return;
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch("/api/python-data");
          if (res.ok && !disposed) setSnapshot(await res.json());
        } catch {
          /* backend unreachable — keep polling */
        }
      }, intervalMs);
    };

    const stopPolling = () => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    };

    const scheduleReconnect = () => {
      if (disposed || retryTimer) return;
      retryTimer = setTimeout(() => {
        retryTimer = null;
        connect();
      }, backoff);
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
    };

    const connect = () => {
      try {
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${proto}//${window.location.host}/ws`);
      } catch {
        startPolling();
        scheduleReconnect();
        return;
      }

      ws.onopen = () => {
        if (disposed) return;
        backoff = 1000;
        stopPolling();
        setConnected(true);
      };

      ws.onmessage = (event) => {
        if (disposed) return;
        try {
          setSnapshot(JSON.parse(event.data));
        } catch {
          /* ignore malformed frames */
        }
      };

      ws.onclose = () => {
        if (disposed) return;
        setConnected(false);
        startPolling();
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (ws) ws.close();
      };
    };

    connect();

    return () => {
      disposed = true;
      if (retryTimer) clearTimeout(retryTimer);
      stopPolling();
      if (ws) ws.close();
    };
  }, [enabled, intervalMs]);

  return { snapshot, connected };
}
