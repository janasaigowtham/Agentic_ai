import { defineConfig, devices } from "@playwright/test";
import { existsSync } from "node:fs";

const isCI = !!process.env.CI;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: isCI ? 1 : 0,
  reporter: isCI ? [["list"], ["html", { open: "never" }]] : [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // This sandbox pins an older Chromium revision than @playwright/test
        // expects for its default headless-shell resolution; launch the
        // pre-installed full browser explicitly instead of downloading one.
        // Harmless outside this sandbox: falls back to Playwright's own
        // resolution when the path doesn't exist (e.g. in CI).
        launchOptions: existsSync("/opt/pw-browsers/chromium")
          ? { executablePath: "/opt/pw-browsers/chromium" }
          : {},
      },
    },
  ],
  webServer: [
    {
      // Own DB file per run so the seeded/synthetic trajectories always
      // start from a clean slate, mock mode by default (no API keys set).
      command:
        "rm -f trh_e2e_test.sqlite3 && TRH_DB_PATH=trh_e2e_test.sqlite3 .venv/bin/python -m uvicorn trh.api.main:app --port 8000",
      cwd: "../backend",
      url: "http://127.0.0.1:8000/api/config",
      reuseExistingServer: !isCI,
      timeout: 30_000,
    },
    {
      command: "npm run dev -- --port 5173 --strictPort",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !isCI,
      timeout: 30_000,
    },
  ],
});
