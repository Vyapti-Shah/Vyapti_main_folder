export default function ProbabilityBar({ label, value }) {
  const pct = Math.round(value * 1000) / 10;
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-track" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label={`${label} probability`}>
        <div className="prob-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="prob-value">{pct.toFixed(1)}%</span>
    </div>
  );
}
