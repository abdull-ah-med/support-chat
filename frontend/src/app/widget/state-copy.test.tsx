import { screen } from "@testing-library/react"
import { describe, expect, test } from "vitest"

import { renderWithProviders } from "@/test/render"

import { ChatStatus } from "./chat-status"

describe("conversation state copy", () => {
  test("reconnecting keeps the transcript status", () => {
    renderWithProviders(<ChatStatus reconnecting={true} />)
    expect(screen.getByText("Reconnecting…")).toBeInTheDocument()
  })
})
