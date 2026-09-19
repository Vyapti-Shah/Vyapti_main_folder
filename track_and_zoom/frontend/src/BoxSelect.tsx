import { useRef, useState } from "react";

/* Drag a box over the paused frame.
 *
 * Emits NORMALISED coordinates (0..1) relative to the VIDEO FRAME, which is
 * what the render needs — it works in the source's native resolution while
 * this is displayed at whatever size the layout gives it.
 *
 * The subtlety that makes this more than a rect subtraction: the <video> is
 * `object-fit: contain`, so the picture is letterboxed inside its element, and
 * this overlay is inset from the bottom to leave the native controls clickable.
 * Normalising against either of those boxes rather than against the displayed
 * PICTURE puts the selection somewhere the user did not point — measured at
 * ~165px horizontally on a 1280x720 clip, enough to seed the tracker on the
 * wrong object entirely. So the content rect is computed from the video's own
 * intrinsic size every time, and every coordinate goes through it.
 */

export interface Box {
  x1: number; y1: number; x2: number; y2: number;   // all 0..1 of the FRAME
}

export interface BoxSelectProps {
  active: boolean;
  box: Box | null;
  onChange: (box: Box | null) => void;
  /** The element being drawn over. Needed for its intrinsic size. */
  getVideo: () => HTMLVideoElement | null;
}

/** Where the picture actually is inside the element, in viewport pixels.
 *  Mirrors what `object-fit: contain` does: scale to fit, centre, letterbox. */
function contentRect(v: HTMLVideoElement): {
  left: number; top: number; width: number; height: number;
} | null {
  const r = v.getBoundingClientRect();
  const nw = v.videoWidth, nh = v.videoHeight;
  // Before metadata loads there is no intrinsic size and no honest mapping.
  if (!nw || !nh || !r.width || !r.height) return null;
  const scale = Math.min(r.width / nw, r.height / nh);
  const w = nw * scale, h = nh * scale;
  return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h };
}

export function BoxSelect({ active, box, onChange, getVideo }: Readonly<BoxSelectProps>) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<{ x: number; y: number } | null>(null);

  /** Pointer position as a fraction of the FRAME, clamped to it. */
  function pos(e: React.PointerEvent): { x: number; y: number } | null {
    const v = getVideo();
    const c = v && contentRect(v);
    if (!c) return null;
    return {
      x: Math.min(1, Math.max(0, (e.clientX - c.left) / c.width)),
      y: Math.min(1, Math.max(0, (e.clientY - c.top) / c.height)),
    };
  }

  function down(e: React.PointerEvent) {
    if (!active) return;
    const p = pos(e);
    if (!p) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(p);
    onChange(null);
  }

  function move(e: React.PointerEvent) {
    if (!active || !dragging) return;
    const p = pos(e);
    if (!p) return;
    onChange({
      x1: Math.min(dragging.x, p.x), y1: Math.min(dragging.y, p.y),
      x2: Math.max(dragging.x, p.x), y2: Math.max(dragging.y, p.y),
    });
  }

  function up() {
    if (!dragging) return;
    setDragging(null);
    // A click with no drag is not a selection — clear rather than submitting a
    // zero-area box the server would reject.
    if (box && (box.x2 - box.x1 < 0.005 || box.y2 - box.y1 < 0.005)) onChange(null);
  }

  /* Drawn through the same mapping it was captured through, expressed relative
   * to this host element — so the rectangle sits exactly where the pointer went
   * and visibly matches what will be sent. If the two used different maths, a
   * mapping bug would be invisible on screen. */
  let style: React.CSSProperties | null = null;
  const v = getVideo();
  const c = v && contentRect(v);
  const host = hostRef.current?.getBoundingClientRect();
  if (box && c && host) {
    style = {
      left: c.left - host.left + box.x1 * c.width,
      top: c.top - host.top + box.y1 * c.height,
      width: (box.x2 - box.x1) * c.width,
      height: (box.y2 - box.y1) * c.height,
    };
  }

  return (
    <div
      ref={hostRef}
      className={`box-select ${active ? "is-active" : ""}`}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
    >
      {style && <div className="box-select__rect" style={style} />}
    </div>
  );
}

export default BoxSelect;
