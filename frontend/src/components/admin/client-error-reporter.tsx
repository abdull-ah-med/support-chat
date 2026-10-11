"use client"

import { useEffect } from "react"

import { getAccessToken, staffRequest } from "@/lib/auth-client"

const reportClientLog = async (payload: {
  event: "ui_window_error" | "ui_unhandled_rejection"
  error_class: string
  line?: number
  column?: number
}) => {
  if (!getAccessToken()) {
    return
  }
  try {
    await staffRequest("/api/logs/client", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  } catch {
    // Client reporting must never break the admin UI.
  }
}

const ERROR_CLASSES = new Set([
  "Error",
  "TypeError",
  "ReferenceError",
  "SyntaxError",
  "RangeError",
  "URIError",
  "EvalError",
  "AggregateError",
  "DOMException",
  "AbortError",
  "NetworkError",
  "UnhandledRejection",
])

export const safeClientErrorMessage = (value: unknown, fallback: string): string =>
  value instanceof Error && ERROR_CLASSES.has(value.name) ? value.name : fallback

const onWindowError = (event: ErrorEvent) => {
  void reportClientLog({
    event: "ui_window_error",
    error_class: safeClientErrorMessage(event.error, "Error"),
    line: event.lineno,
    column: event.colno,
  })
}

const onUnhandledRejection = (event: PromiseRejectionEvent) => {
  void reportClientLog({
    event: "ui_unhandled_rejection",
    error_class: safeClientErrorMessage(event.reason, "UnhandledRejection"),
  })
}

export const installClientErrorReporting = () => {
  window.addEventListener("error", onWindowError)
  window.addEventListener("unhandledrejection", onUnhandledRejection)
  return () => {
    window.removeEventListener("error", onWindowError)
    window.removeEventListener("unhandledrejection", onUnhandledRejection)
  }
}

export const ClientErrorReporter = () => {
  useEffect(() => installClientErrorReporting(), [])
  return null
}
