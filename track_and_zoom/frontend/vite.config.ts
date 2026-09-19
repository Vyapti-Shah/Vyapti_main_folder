import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `/api` is proxied to the backend in dev so the app is same-origin in both
// dev and production (nginx does the same proxying in the container) — that
// keeps one relative-URL API client working in both, with no CORS special case.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: { "/api": { target: process.env.TZ_API ?? "http://localhost:18100", changeOrigin: true } },
  },
});
