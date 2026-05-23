"use client";

import { useEffect, useRef } from "react";
import type { Data, Layout, Config } from "plotly.js-dist-min";

type PlotlyLib = typeof import("plotly.js-dist-min");

function readCssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function useThemedPlot(
  build: (Plotly: PlotlyLib, theme: PlotTheme) => { data: Data[]; layout: Partial<Layout>; config?: Partial<Config> },
) {
  const ref = useRef<HTMLDivElement | null>(null);
  const buildRef = useRef(build);
  buildRef.current = build;

  useEffect(() => {
    let disposed = false;
    let cleanup: (() => void) | null = null;

    (async () => {
      const Plotly = (await import("plotly.js-dist-min")).default;
      if (disposed || !ref.current) return;

      const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      const theme: PlotTheme = {
        fg: readCssVar("--fg", isDark ? "#e9edf1" : "#1a1a1a"),
        muted: readCssVar("--muted", isDark ? "#9aa3ad" : "#5b6470"),
        rule: readCssVar("--rule", isDark ? "#2a2f35" : "#e6e8eb"),
        bg: readCssVar("--bg", isDark ? "#14171a" : "#ffffff"),
      };

      const { data, layout, config } = buildRef.current(Plotly, theme);
      const baseLayout: Partial<Layout> = {
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: {
          family:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          color: theme.fg,
          size: 13,
        },
        margin: { l: 56, r: 20, t: 36, b: 52 },
        legend: { bgcolor: "rgba(0,0,0,0)", font: { color: theme.fg } },
        hoverlabel: { bgcolor: theme.bg, bordercolor: theme.rule, font: { color: theme.fg } },
      };

      const mergedLayout: Partial<Layout> = {
        ...baseLayout,
        ...layout,
        xaxis: {
          gridcolor: theme.rule,
          zerolinecolor: theme.rule,
          linecolor: theme.rule,
          tickcolor: theme.rule,
          color: theme.muted,
          ...(layout.xaxis ?? {}),
        },
        yaxis: {
          gridcolor: theme.rule,
          zerolinecolor: theme.rule,
          linecolor: theme.rule,
          tickcolor: theme.rule,
          color: theme.muted,
          ...(layout.yaxis ?? {}),
        },
      };

      const finalConfig: Partial<Config> = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        ...config,
      };

      await Plotly.newPlot(ref.current, data, mergedLayout, finalConfig);

      const el = ref.current;
      cleanup = () => {
        if (el) Plotly.purge(el);
      };
    })();

    return () => {
      disposed = true;
      if (cleanup) cleanup();
    };
  }, []);

  return ref;
}

type PlotTheme = { fg: string; muted: string; rule: string; bg: string };

const palette = {
  omp: "#d1495b",
  fista: "#0a4d8c",
  admm: "#2e933c",
  seq: "#9b59b6",
  joint: "#0a4d8c",
  ista: "#d1495b",
  lista: "#2e933c",
};

// ───── Rate-distortion ─────
const rd = {
  deltas: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
  omp: { mean: [15.46, 17.42, 19.15, 20.42, 21.73, 23.5, 24.89], std: [4.44, 4.33, 4.04, 3.7, 3.2, 3.0, 2.97] },
  fista: { mean: [16.67, 19.39, 21.73, 23.64, 25.06, 26.43, 27.57], std: [4.85, 4.49, 4.28, 4.09, 3.93, 3.82, 3.52] },
  admm: { mean: [11.23, 17.62, 21.27, 23.47, 24.98, 26.37, 27.45], std: [3.07, 4.78, 4.61, 4.36, 4.0, 3.89, 3.65] },
};

export function RateDistortionPlot() {
  const ref = useThemedPlot(() => {
    const trace = (key: "omp" | "fista" | "admm", name: string, color: string): Data => ({
      x: rd.deltas,
      y: rd[key].mean,
      name,
      type: "scatter",
      mode: "lines+markers",
      line: { color, width: 2.5 },
      marker: { color, size: 7 },
      error_y: { type: "data", array: rd[key].std, visible: true, color, thickness: 1, width: 4 },
      hovertemplate: `<b>${name}</b><br>δ = %{x}<br>PSNR = %{y:.2f} dB<extra></extra>`,
    });
    return {
      data: [
        trace("omp", "OMP", palette.omp),
        trace("fista", "FISTA", palette.fista),
        trace("admm", "ADMM", palette.admm),
      ],
      layout: {
        title: { text: "PSNR vs measurement rate δ = M/N", font: { size: 14 } },
        xaxis: { title: { text: "measurement rate δ" }, dtick: 0.1 },
        yaxis: { title: { text: "PSNR (dB)" } },
      },
    };
  });
  return <div ref={ref} className="plot" />;
}

// ───── LISTA bar chart ─────
export function ListaPlot() {
  const ref = useThemedPlot(() => ({
    data: [
      {
        x: ["ISTA", "FISTA", "LISTA"],
        y: [0.4667, 0.3369, 0.0501],
        type: "bar",
        marker: { color: [palette.ista, palette.fista, palette.lista] },
        text: ["0.467", "0.337", "0.050"],
        textposition: "outside",
        hovertemplate: "<b>%{x}</b><br>NMSE = %{y:.4f}<extra></extra>",
      } as Data,
    ],
    layout: {
      title: { text: "Validation NMSE at K = 10 (lower is better)", font: { size: 14 } },
      xaxis: { title: { text: "solver" } },
      yaxis: { title: { text: "validation NMSE (log)" }, type: "log" },
      showlegend: false,
    },
  }));
  return <div ref={ref} className="plot" />;
}

// ───── Joint vs Sequential aggregate ─────
const jvs = {
  deltas: [0.2, 0.3, 0.4, 0.5, 0.6],
  seq: { mean: [10.717, 11.648, 12.513, 13.593, 13.402], std: [3.393, 3.754, 2.335, 2.421, 2.385] },
  joint: { mean: [9.032, 10.271, 11.585, 12.759, 12.937], std: [1.565, 1.503, 1.641, 2.003, 1.998] },
};

export function JointVsSequentialPlot() {
  const ref = useThemedPlot(() => {
    const band = (key: "seq" | "joint", name: string, color: string): Data[] => {
      const upper = jvs[key].mean.map((m, i) => m + jvs[key].std[i]);
      const lower = jvs[key].mean.map((m, i) => m - jvs[key].std[i]);
      return [
        {
          x: [...jvs.deltas, ...[...jvs.deltas].reverse()],
          y: [...upper, ...[...lower].reverse()],
          fill: "toself",
          fillcolor: hexToRgba(color, 0.12),
          line: { color: "transparent" },
          name: `${name} ±1σ`,
          hoverinfo: "skip",
          showlegend: false,
          type: "scatter",
        } as Data,
        {
          x: jvs.deltas,
          y: jvs[key].mean,
          name,
          type: "scatter",
          mode: "lines+markers",
          line: { color, width: 2.5 },
          marker: { color, size: 7 },
          hovertemplate: `<b>${name}</b><br>δ = %{x}<br>PSNR = %{y:.2f} dB<extra></extra>`,
        } as Data,
      ];
    };
    return {
      data: [
        ...band("seq", "Sequential (CS → illum-correct)", palette.seq),
        ...band("joint", "Joint-CS", palette.joint),
      ],
      layout: {
        title: { text: "Aggregate PSNR under 8× illumination gradient", font: { size: 14 } },
        xaxis: { title: { text: "measurement rate δ" }, dtick: 0.1 },
        yaxis: { title: { text: "PSNR (dB)" } },
      },
    };
  });
  return <div ref={ref} className="plot" />;
}

// ───── Per-scene Δ grouped bars ─────
const scenes = ["cameraman", "astronaut", "coins", "page", "moon"];
const psnrDiff: Record<string, number[]> = {
  "0.20": [2.87, 0.54, -1.02, -0.79, -10.03],
  "0.30": [4.65, 0.18, -1.73, -0.48, -9.51],
  "0.40": [1.13, 1.84, 0.39, -0.23, -7.76],
  "0.50": [1.08, 2.31, 0.74, 0.74, -9.05],
  "0.60": [3.07, 2.21, 0.32, 0.43, -8.37],
};
const psColors = ["#9ec5e8", "#6aa3d4", "#0a4d8c", "#d1495b", "#7b1d2f"];

export function PerScenePlot() {
  const ref = useThemedPlot((_Plotly, theme) => ({
    data: Object.keys(psnrDiff).map((d, i) => ({
      x: scenes,
      y: psnrDiff[d],
      name: `δ = ${d}`,
      type: "bar",
      marker: { color: psColors[i] },
      hovertemplate: `<b>%{x}</b><br>δ = ${d}<br>Δ = %{y:.2f} dB<extra></extra>`,
    })) as Data[],
    layout: {
      title: { text: "Per-scene PSNR gain Δ = joint − sequential (dB)", font: { size: 14 } },
      barmode: "group",
      xaxis: { title: { text: "scene" } },
      yaxis: {
        title: { text: "Δ PSNR (dB)" },
        zeroline: true,
        zerolinewidth: 1.5,
        zerolinecolor: theme.muted,
      },
    },
  }));
  return <div ref={ref} className="plot tall" />;
}
