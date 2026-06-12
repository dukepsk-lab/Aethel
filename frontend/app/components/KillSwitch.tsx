"use client";

import { useState } from "react";
import { API } from "../page";
import styles from "./KillSwitch.module.css";

export function KillSwitch({ tripped, onTrip }: { tripped: boolean; onTrip: () => void }) {
  const [confirming, setConfirming] = useState(false);

  const handleClick = async () => {
    if (tripped) return;
    if (!confirming) { setConfirming(true); return; }
    await fetch(`${API}/risk/kill-switch`, { method: "POST" });
    setConfirming(false);
    onTrip();
  };

  return (
    <div className={styles.wrap}>
      {confirming && !tripped && (
        <span className={styles.confirm}>
          Confirm? &nbsp;
          <button className={styles.cancel} onClick={() => setConfirming(false)}>Cancel</button>
        </span>
      )}
      <button
        className={`${styles.btn} ${tripped ? styles.tripped : confirming ? styles.armed : ""}`}
        onClick={handleClick}
        disabled={tripped}
      >
        {tripped ? "🛑 KILLED" : confirming ? "⚠ CONFIRM KILL" : "🛑 KILL SWITCH"}
      </button>
    </div>
  );
}
