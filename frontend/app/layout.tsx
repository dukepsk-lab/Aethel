export const metadata = { title: "NEXMIND — Aethel Command Center" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "ui-monospace, monospace", background: "#0b0e14", color: "#e6e6e6", margin: 0 }}>
        {children}
      </body>
    </html>
  );
}
