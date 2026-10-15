# Forensic Video Report

    docker compose up --build

Open http://localhost:8000, upload videos (and optionally a reference photo), and a report
styled like the reference screenshot is built and updates live while frames are extracted.

- Metadata comes from `ffprobe`; file times are the times of the uploaded copy inside the container
  (the original device's Created/Modified times are not preserved by browser uploads).
- "Recorded" is the container's `creation_time` tag, if present.
- Thumbnails are sampled at fixed intervals (`FRAME_INTERVAL_SEC`, capped at `MAX_FRAMES` per video
  in `docker-compose.yml`). They are NOT face-matched; the reference photo is shown for context only.
- SHA-256 of each uploaded file is included in the report.
- Use the "Print / save as PDF" link for a PDF export.
- Data persists in the `report_data` Docker volume.
