// Color alone never carries the meaning: each verdict also gets an icon and text label.
const VERDICTS = {
  REAL: { cls: "verdict-real", icon: "✓", label: "REAL" },
  FAKE: { cls: "verdict-fake", icon: "⚠", label: "FAKE" },
  INCONCLUSIVE: { cls: "verdict-inconclusive", icon: "?", label: "INCONCLUSIVE" },
};

export default function VerdictBadge({ prediction }) {
  const v = VERDICTS[prediction] ?? VERDICTS.INCONCLUSIVE;
  return (
    <span className={`verdict ${v.cls}`}>
      <span aria-hidden="true">{v.icon}</span> {v.label}
    </span>
  );
}
