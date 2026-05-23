import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://snapshot-raw-enhance.vercel.app"),
  title:
    "Joint-CS: Simultaneous Illumination Normalization and Sparse Image Compression via Unfolded Recovery",
  description:
    "Joint-CS: a single inverse problem in DCT-domain reflectance coefficients and a smooth per-column illumination field, solved by block-coordinate alternation with a learned LISTA c-step.",
  openGraph: {
    title:
      "Joint-CS: Simultaneous Illumination Normalization and Sparse Image Compression via Unfolded Recovery",
    description:
      "ENGS 109 final project — Farhan Sadeek, Thayer School of Engineering, Dartmouth College.",
    images: ["/figures/rate_distortion.png"],
  },
  twitter: { card: "summary_large_image" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <Script id="mathjax-config" strategy="beforeInteractive">{`
          window.MathJax = {
            tex: {
              inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
              displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
              processEscapes: true
            },
            svg: { fontCache: 'global' }
          };
        `}</Script>
        <Script
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
          strategy="afterInteractive"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
