import ProbabilityBar from "./ProbabilityBar.jsx";
import VerdictBadge from "./VerdictBadge.jsx";

function FrameList({ scores }) {
  return (
    <details className="frame-list">
      <summary>Per-frame scores ({scores.length})</summary>
      <ol>
        {scores.map((p, i) => (
          <li key={i}>
            <span className="frame-idx">#{i + 1}</span>
            <div className="prob-track prob-track--small" aria-hidden="true">
              <div className="prob-fill" style={{ width: `${p * 100}%` }} />
            </div>
            <span className="frame-score">{(p * 100).toFixed(1)}% fake</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

export default function ResultPanel({ result }) {
  const {
    prediction,
    confidence,
    probabilities,
    inference_ms,
    evaluation,
    frames_analyzed,
    frame_scores,
    aggregation,
    note,
  } = result;

  return (
    <div className="result" role="status">
      <div className="result-head">
        <VerdictBadge prediction={prediction} />
        <span className="confidence">
          {confidence == null ? "Confidence: —" : `Confidence: ${(confidence * 100).toFixed(1)}%`}
        </span>
      </div>

      <ProbabilityBar label="Real" value={probabilities.real} />
      <ProbabilityBar label="Fake" value={probabilities.fake} />

      <dl className="result-meta">
        <div>
          <dt>Inference time</dt>
          <dd>{inference_ms} ms</dd>
        </div>
        {evaluation && (
          <div>
            <dt>Dataset evaluation accuracy</dt>
            <dd>
              {(evaluation.accuracy * 100).toFixed(1)}% ({evaluation.dataset}, n={evaluation.samples})
            </dd>
          </div>
        )}
        {frames_analyzed != null && (
          <div>
            <dt>Frames analyzed</dt>
            <dd>{frames_analyzed}</dd>
          </div>
        )}
      </dl>

      {aggregation && <p className="aggregation-note">{aggregation.description}</p>}
      {frame_scores?.length > 0 && <FrameList scores={frame_scores} />}

      <p className="note">{note}</p>
    </div>
  );
}
