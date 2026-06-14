"use client";

import styles from "./ModelStatus.module.css";

const SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"];

interface ArtifactInfo {
  loaded: boolean;
  version: string | null;
}

interface SymbolModels {
  venus: ArtifactInfo;
  helios: ArtifactInfo;
}

export function ModelStatus({ data }: { data: Record<string, SymbolModels> | undefined }) {
  return (
    <section className={styles.card}>
      <div className={styles.header}>Model Status</div>
      <div className={styles.grid}>
        {SYMBOLS.map((sym) => {
          const info = data?.[sym];
          const venus = info?.venus;
          const helios = info?.helios;
          return (
            <div key={sym} className={styles.row}>
              <span className={styles.sym}>{sym}</span>
              <ModelPill label="Venus" info={venus} />
              <ModelPill label="Helios" info={helios} />
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ModelPill({ label, info }: { label: string; info: ArtifactInfo | undefined }) {
  if (!info) {
    return <span className={`${styles.pill} ${styles.missing}`}>{label} —</span>;
  }
  if (!info.loaded) {
    return <span className={`${styles.pill} ${styles.missing}`}>{label} missing</span>;
  }
  const ver = info.version ?? "unknown";
  // version format: YYYYMMDD-HHMM — show as MMM DD HH:MM
  let verShort = ver;
  const m = ver.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})$/);
  if (m) {
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    verShort = `${months[parseInt(m[2]) - 1]} ${m[3]} ${m[4]}:${m[5]}`;
  }
  return (
    <span className={`${styles.pill} ${styles.loaded}`} title={ver}>
      {label} <span className={styles.ver}>{verShort}</span>
    </span>
  );
}
