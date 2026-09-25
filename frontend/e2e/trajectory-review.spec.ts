import { test, expect } from "@playwright/test";

// Full pipeline is Capture -> Trajectory Assembler -> Evidence Assembly ->
// Orient -> Specialists -> Aggregator -> Fact-check -> human Gate. These
// tests drive it the way a reviewer actually would: through the UI, against
// the real FastAPI backend and its SQLite store, in mock judge mode (no API
// keys). Tests run serially (see playwright.config.ts) since later tests
// build on state the earlier ones leave behind.

test.describe("Trajectory Review Harness", () => {
  test("lists all trajectory sources and shows the mock mode badge", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".mode-badge")).toHaveText("MOCK");
    // 1 hand-recorded fixture + 4 executor-generated synthetic scenarios.
    await expect(page.locator("table.record-table tbody tr")).toHaveCount(5);
  });

  test("runs a review on the seeded fixture and surfaces every seeded finding", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto("/");
    const seededRow = page.locator("table.record-table tbody tr", { hasText: "seeded fixture" });
    await seededRow.getByRole("button", { name: "Run review" }).click();

    await expect(page.locator(".case-summary")).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(".step-row")).toHaveCount(17);

    // D1 (duplicate DB call) and N3 (undeclared prompt reference) both
    // highlight in the step viewer.
    await expect(page.locator(".step-row.declared-mismatch")).toHaveCount(2);

    await expect(page.locator(".findings-panel h2")).toContainText("Findings (17, none dropped)");
    // N3: pmi_ddn_verdict_synthesizer reads the raw loan record undeclared.
    await expect(page.locator(".finding.severity-critical")).toHaveCount(1);

    // D1 (duplicate call), N3 (undeclared ref), N2 (shared prompt block,
    // found cross-step by the Aggregator).
    await expect(page.locator(".recommendation")).toHaveCount(3);

    expect(consoleErrors).toEqual([]);
  });

  test("approving a recommendation updates its status without a reload", async ({ page }) => {
    await page.goto("/");
    const seededRow = page.locator("table.record-table tbody tr", { hasText: "seeded fixture" });
    await seededRow.getByRole("button", { name: "Run review" }).click();
    await expect(page.locator(".gate-panel")).toBeVisible({ timeout: 15_000 });

    const firstRec = page.locator(".recommendation").first();
    await expect(firstRec).toHaveClass(/status-proposed/);

    await firstRec.getByRole("button", { name: "Approve" }).click();
    await expect(firstRec).toHaveClass(/status-approved/);
    await expect(firstRec.getByRole("button", { name: "Approve" })).toBeDisabled();
  });

  test("an executor-generated synthetic trajectory reviews cleanly end to end", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto("/");
    const synthRow = page.locator("table.record-table tbody tr", {
      hasText: "Synthetic loan 1234567890",
    });
    await synthRow.getByRole("button", { name: "Run review" }).click();

    await expect(page.locator(".case-summary")).toBeVisible({ timeout: 15_000 });
    // one fewer step than the hand-recorded fixture: no scripted duplicate call.
    await expect(page.locator(".step-row")).toHaveCount(16);
    // Evidence Assembly still finds the same-class undeclared reference
    // organically on freshly executed data.
    await expect(page.locator(".step-row.declared-mismatch")).toHaveCount(1);

    expect(consoleErrors).toEqual([]);
  });
});
