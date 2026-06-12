"use client";

import { useEffect, useRef } from "react";
import styles from "./EquityChart.module.css";

export function EquityChart({ trades }: { trades: any[] | undefined }) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const seriesRef = useRef<any>(null);

  useEffect(() => {
    if (!ref.current) return;
    let chart: any;
    import("lightweight-charts").then(({ createChart, LineStyle }) => {
      chart = createChart(ref.current!, {
        layout: { background: { color: "transparent" }, textColor: "#7d8590" },
        grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
        crosshair: { vertLine: { color: "#58a6ff55" }, horzLine: { color: "#58a6ff55" } },
        rightPriceScale: { borderColor: "#21262d" },
        timeScale: { borderColor: "#21262d", timeVisible: true },
        handleScroll: true,
        handleScale: true,
      });
      const series = chart.addLineSeries({
        color: "#58a6ff",
        lineWidth: 2,
        crosshairMarkerVisible: true,
        priceLineVisible: false,
      });
      chartRef.current = chart;
      seriesRef.current = series;

      const ro = new ResizeObserver(() => {
        if (ref.current) chart.resize(ref.current.clientWidth, ref.current.clientHeight);
      });
      ro.observe(ref.current!);
      return () => { ro.disconnect(); chart.remove(); };
    });
    return () => { if (chartRef.current) chartRef.current.remove(); };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || !trades?.length) return;
    let equity = 10_000;
    const points = trades
      .slice()
      .sort((a: any, b: any) => new Date(a.closed_at).getTime() - new Date(b.closed_at).getTime())
      .map((t: any) => {
        equity += t.profit;
        return {
          time: Math.floor(new Date(t.closed_at).getTime() / 1000) as any,
          value: parseFloat(equity.toFixed(2)),
        };
      });
    if (points.length) seriesRef.current.setData(points);
  }, [trades]);

  return (
    <div className={styles.wrap}>
      {(!trades || trades.length === 0) && (
        <div className={styles.empty}>No closed trades yet — shadow mode active</div>
      )}
      <div ref={ref} className={styles.chart} />
    </div>
  );
}
