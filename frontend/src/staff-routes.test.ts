import { NextRequest } from "next/server"
import { describe, expect, test } from "vitest"

import config from "../next.config"
import { proxy } from "./proxy"

const NON_STAFF_HOSTS = ["widget.localhost:3000", "host.localhost:3000"]

describe("non-staff hosts cannot reach staff surfaces", () => {
  test.each(NON_STAFF_HOSTS)("%s cannot register a staff push worker or access device APIs", async (host) => {
    const paths = [
      "/staff-push-sw.js",
      "/manifest.webmanifest",
      "/api/notifications/push/config",
      "/api/notifications/push/subscriptions",
    ]
    const responses = await Promise.all(
      paths.map((path) => proxy(new NextRequest(`http://${host}${path}`, { headers: { host } }))),
    )
    for (const response of responses) expect(response.status).toBe(404)
  })

  test.each(NON_STAFF_HOSTS)("%s cannot access workspace search", async (host) => {
    const response = await proxy(
      new NextRequest(`http://${host}/api/search`, { method: "POST", headers: { host } }),
    )
    expect(response.status).toBe(404)
  })

  test.each(NON_STAFF_HOSTS)("%s cannot access staff notifications", async (host) => {
    const response = await proxy(
      new NextRequest(`http://${host}/api/notifications/preferences`, { headers: { host } }),
    )
    expect(response.status).toBe(404)
  })
})

describe("staff API rewrites", () => {
  const API = "http://127.0.0.1:8000"
  const sources = [
    "/api/search",
    "/api/notifications",
    "/api/notifications/:path*",
    "/api/handoffs/:path*",
    "/api/logs/:path*",
    "/api/status",
    "/api/knowledge-gaps",
    "/api/knowledge-gaps/:path*",
    "/api/visitor-blocks",
    "/api/visitor-blocks/:path*",
  ]

  test("forwards staff routes to the backend", async () => {
    const rules = await config.rewrites!()
    const list = Array.isArray(rules)
      ? rules
      : [...(rules.beforeFiles ?? []), ...(rules.afterFiles ?? []), ...(rules.fallback ?? [])]
    for (const source of sources) {
      expect(list.find((rule) => rule.source === source)?.destination).toBe(`${API}${source}`)
    }
  })
})
