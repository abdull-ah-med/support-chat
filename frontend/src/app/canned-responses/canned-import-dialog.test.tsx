/* oxlint-disable react-perf/jsx-no-new-function-as-prop, eslint/max-lines-per-function -- Test doubles and preview stubs are created per case. */

import { fireEvent, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, test, vi } from "vitest"

import { setAccessToken } from "@/lib/auth-client"
import { renderWithProviders } from "@/test/render"

import { ImportDialog } from "./canned-import-dialog"

const noopImported = async () => undefined
const SHOP_SITE = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
const SUPPORT_SITE = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
const BLOG_SITE = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
const SITES = [
  { id: SHOP_SITE, name: "Shop" },
  { id: SUPPORT_SITE, name: "Support" },
]
const SHOP_BLOG_SITES = [
  { id: SHOP_SITE, name: "Shop" },
  { id: BLOG_SITE, name: "Blog" },
]

const RETURNS_BODY = "Returns are accepted within 30 days."
const PACKAGES_BODY = "Packages start at $19.95."
const UPDATED_HOURS = "We are open from eight to six on weekdays."
const FILL_IN_BODY = "Your request was received on [DATE/TIME]."

const jsonResponse = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => body,
})

const previewRow = (overrides: Record<string, unknown>) => ({
  action: "create",
  reason: null,
  livechat_id: 1,
  group: 6,
  site_id: SHOP_SITE,
  shortcut: "hours",
  aliases: [],
  bot_eligible: true,
  disable_reason: null,
  suggestion_event: null,
  excerpt: "We are open from nine to five on weekdays.",
  ...overrides,
})

const renderImport = (sites = SITES) =>
  renderWithProviders(
    <ImportDialog open onClose={vi.fn()} onImported={noopImported} sites={sites} />,
  )

const chooseCsv = async () => {
  const user = userEvent.setup()
  const file = new File(["id,text,tags,group"], "canned.csv", { type: "text/csv" })
  await user.upload(screen.getByLabelText("LiveChat canned responses CSV"), file)
  return user
}

describe("canned import dialog", () => {
  beforeEach(() => {
    setAccessToken("staff-token")
    vi.unstubAllGlobals()
  })

  test("asks for a CSV without listing LiveChat group numbers", () => {
    renderImport()

    expect(screen.getByText("Import LiveChat canned responses")).toBeInTheDocument()
    expect(screen.getByLabelText("LiveChat canned responses CSV")).toBeInTheDocument()
    expect(screen.queryByText(/Group 0/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/id, text, tags, group/i)).not.toBeInTheDocument()
  })

  test("closes when the dimmed area outside the dialog is clicked", async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    renderWithProviders(
      <ImportDialog open onClose={onClose} onImported={noopImported} sites={SITES} />,
    )
    expect(screen.getByText("Import LiveChat canned responses")).toBeInTheDocument()

    const overlay = document.querySelector("[data-slot='dialog-overlay']")
    if (!overlay) throw new Error("expected dialog overlay")
    await user.click(overlay)

    await waitFor(() => expect(onClose).toHaveBeenCalled())
  })

  test("lets staff drag the corner to enlarge the import dialog", async () => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 1400 })
    Object.defineProperty(window, "innerHeight", { configurable: true, value: 1100 })
    renderImport()
    const dialog = screen.getByRole("dialog", { name: "Import LiveChat canned responses" })
    const handle = screen.getByRole("button", { name: "Resize dialog" })
    expect(Number.parseFloat(dialog.style.width || "0")).toBe(768)
    expect(Number.parseFloat(dialog.style.height || "0")).toBe(720)

    fireEvent.pointerDown(handle, { button: 0, clientX: 700, clientY: 500 })
    window.dispatchEvent(
      new PointerEvent("pointermove", { clientX: 860, clientY: 620, buttons: 1, bubbles: true }),
    )
    window.dispatchEvent(
      new PointerEvent("pointerup", { clientX: 860, clientY: 620, bubbles: true }),
    )

    // Avg of (768+160)/768 and (720+120)/720 is 1.1875 → 912×855 with locked aspect.
    await waitFor(() => {
      expect(Number.parseFloat(dialog.style.width || "0")).toBe(912)
      expect(Number.parseFloat(dialog.style.height || "0")).toBe(855)
    })
  })
})

const committedDecisions = (fetchMock: {
  mock: { calls: [RequestInfo | URL, RequestInit?][] }
}) => {
  const commit = fetchMock.mock.calls.find(
    (call) => String(call[0]) === "/api/canned-replies/import",
  )
  if (!commit) throw new Error("expected import commit")
  const body = commit[1]?.body
  if (!(body instanceof FormData)) throw new Error("expected multipart import")
  return JSON.parse(String(body.get("decisions")))
}

const prompt = (group: number) =>
  `Choose a website for LiveChat group ${group}, or discard those responses.`

const stubPreview = (rows: unknown[], commit = { created: 1, updated: 0, skipped: 0 }) => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = String(input)
    if (url === "/api/canned-replies/import/preview") {
      return jsonResponse({ created: 0, updated: 0, skipped: 0, rows })
    }
    if (url === "/api/canned-replies/import") return jsonResponse(commit)
    return jsonResponse({ detail: "missing" }, 404)
  })
  vi.stubGlobal("fetch", fetchMock)
  return fetchMock
}

describe("canned import preview decisions", () => {
  beforeEach(() => {
    setAccessToken("staff-token")
    vi.unstubAllGlobals()
  })

  test("offers an unmapped group every website or discard instead of hiding the copy", async () => {
    stubPreview([
      previewRow({
        action: "unmapped",
        livechat_id: 501,
        group: 5,
        site_id: null,
        shortcut: "returns",
        excerpt: RETURNS_BODY,
      }),
    ])
    renderImport()
    const user = await chooseCsv()
    await user.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByText("LiveChat group 5")).toBeInTheDocument()
    expect(screen.getByText(RETURNS_BODY)).toBeInTheDocument()
    expect(screen.getByLabelText("Add LiveChat group 5 responses to")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Discard LiveChat group 5 responses" }),
    ).toBeInTheDocument()
    await user.click(screen.getByLabelText("Add LiveChat group 5 responses to"))
    expect(await screen.findByRole("option", { name: "Shop" })).toBeInTheDocument()
    expect(screen.getByRole("option", { name: "Support" })).toBeInTheDocument()
    expect(screen.getByRole("option", { name: "General — all websites" })).toBeInTheDocument()
  })

  test("requires a website choice for every unmapped group before importing", async () => {
    const fetchMock = stubPreview(
      [
        previewRow({ livechat_id: 200, group: 6, site_id: SHOP_SITE }),
        previewRow({
          action: "unmapped",
          livechat_id: 410,
          group: 4,
          site_id: null,
          shortcut: "packages",
          excerpt: PACKAGES_BODY,
        }),
        previewRow({
          action: "unmapped",
          livechat_id: 501,
          group: 5,
          site_id: null,
          shortcut: "returns",
          excerpt: RETURNS_BODY,
        }),
      ],
      { created: 3, updated: 0, skipped: 0 },
    )
    renderImport(SHOP_BLOG_SITES)
    const user = await chooseCsv()
    await user.click(screen.getByRole("button", { name: "Preview" }))
    await screen.findByText(PACKAGES_BODY)

    expect(screen.getByText(prompt(4))).toBeInTheDocument()
    expect(screen.getByText(prompt(5))).toBeInTheDocument()
    expect(screen.queryByText(prompt(6))).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Import responses" })).toBeDisabled()

    await user.click(screen.getByLabelText("Add LiveChat group 4 responses to"))
    await user.click(await screen.findByRole("option", { name: "Blog" }))
    expect(screen.queryByText(prompt(4))).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Import responses" })).toBeDisabled()

    await user.click(screen.getByLabelText("Add LiveChat group 5 responses to"))
    await user.click(await screen.findByRole("option", { name: "Blog" }))
    expect(screen.getByRole("button", { name: "Import responses" })).toBeEnabled()
    await user.click(screen.getByRole("button", { name: "Import responses" }))

    await waitFor(() =>
      expect(committedDecisions(fetchMock)).toEqual({
        discard_ids: [],
        remap_groups: { "4": BLOG_SITE, "5": BLOG_SITE, "6": SHOP_SITE },
      }),
    )
  })

  test("discarding a group sends its ids as discarded and no site", async () => {
    const fetchMock = stubPreview([
      previewRow({
        action: "unmapped",
        livechat_id: 501,
        group: 5,
        site_id: null,
        shortcut: "returns",
        excerpt: RETURNS_BODY,
      }),
    ])
    renderImport()
    const user = await chooseCsv()
    await user.click(screen.getByRole("button", { name: "Preview" }))
    await user.click(
      await screen.findByRole("button", { name: "Discard LiveChat group 5 responses" }),
    )
    await user.click(screen.getByRole("button", { name: "Import responses" }))

    await waitFor(() =>
      expect(committedDecisions(fetchMock)).toEqual({ discard_ids: [501], remap_groups: {} }),
    )
  })

  test("lets staff keep or update selected duplicate responses", async () => {
    const fetchMock = stubPreview(
      [
        previewRow({
          action: "update",
          livechat_id: 200,
          shortcut: "hours",
          excerpt: UPDATED_HOURS,
        }),
      ],
      { created: 0, updated: 0, skipped: 1 },
    )
    renderImport()
    const user = await chooseCsv()
    await user.click(screen.getByRole("button", { name: "Preview" }))

    const checkbox = await screen.findByRole("checkbox", { name: "Update #hours" })
    expect(checkbox).toBeChecked()
    expect(screen.getAllByText(UPDATED_HOURS).length).toBeGreaterThan(0)
    await user.click(screen.getByRole("button", { name: "Select none" }))
    expect(checkbox).not.toBeChecked()
    await user.click(screen.getByRole("button", { name: "Import responses" }))

    await waitFor(() =>
      expect(committedDecisions(fetchMock)).toEqual({
        discard_ids: [200],
        remap_groups: { "6": SHOP_SITE },
      }),
    )
  })

  test("lists General responses and marks agent-disabled ones with a pill", async () => {
    const generalBody = "Can I help you with anything else?"
    stubPreview([
      previewRow({
        livechat_id: 14,
        group: 0,
        site_id: null,
        shortcut: "help",
        excerpt: generalBody,
        bot_eligible: false,
        disable_reason: "Greeting or idle prompt — not available to the bot.",
      }),
      previewRow({
        livechat_id: 310,
        group: 0,
        site_id: null,
        shortcut: "received",
        excerpt: FILL_IN_BODY,
        bot_eligible: false,
        disable_reason: "Has fill-in fields — not available to the bot.",
      }),
    ])
    renderImport()
    const user = await chooseCsv()
    await user.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByText("LiveChat group 0")).toBeInTheDocument()
    expect(screen.getByText(generalBody)).toBeInTheDocument()
    expect(screen.getByText(FILL_IN_BODY)).toBeInTheDocument()
    expect(screen.getAllByText("Agent Disabled")).toHaveLength(2)
  })

  test("shows rows the server skipped as could not import", async () => {
    stubPreview([
      previewRow({
        action: "skip",
        livechat_id: 502,
        group: 5,
        site_id: null,
        shortcut: "",
        excerpt: RETURNS_BODY,
        reason: "Message is empty.",
      }),
    ])
    renderImport()
    const user = await chooseCsv()
    await user.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByText("Could not import")).toBeInTheDocument()
    expect(screen.getByText("Message is empty.")).toBeInTheDocument()
    expect(screen.queryByLabelText("Add LiveChat group 5 responses to")).not.toBeInTheDocument()
  })
})
