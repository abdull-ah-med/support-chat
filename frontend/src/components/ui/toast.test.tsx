import { act, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { Toaster, toast } from "@/components/ui/toast"

describe("Toaster", () => {
  it("anchors the viewport to the top and swipes up or right", async () => {
    render(<Toaster />)
    act(() => {
      toast.add({ title: "Saved", description: "All good" })
    })

    const title = await screen.findByText("Saved")
    const viewport = document.querySelector('[data-slot="toast-viewport"]') as HTMLElement
    expect(viewport.className).toContain("top-[max(1rem,env(safe-area-inset-top))]")
    expect(viewport.className).not.toContain("bottom-")

    const root = title.closest('[data-slot="toast"]') as HTMLElement
    expect(root.className).toContain("top-0")
    expect(root.className).toContain("origin-top")
    expect(root.className).not.toContain("origin-bottom")
    expect(root.className).toContain("after:bottom-full")
    expect(root.className).not.toContain("swipe-direction=down")
  })
})
