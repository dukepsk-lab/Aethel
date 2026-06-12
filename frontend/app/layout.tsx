import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEXMIND — Aethel Command Center",
  description: "Real-time monitoring for the Aethel trading system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
