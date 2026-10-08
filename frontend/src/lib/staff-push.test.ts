import { afterEach, expect, test, vi } from "vitest"

import {
  disablePush,
  enablePush,
  hasActivePush,
  previewSystemSound,
  syncExistingPush,
} from "./staff-push"

const serviceWorkerDescriptor = Object.getOwnPropertyDescriptor(navigator, "serviceWorker")
afterEach(() => {
  localStorage.clear()
  vi.unstubAllGlobals()
  if (serviceWorkerDescriptor)
    Object.defineProperty(navigator, "serviceWorker", serviceWorkerDescriptor)
  else Reflect.deleteProperty(navigator, "serviceWorker")
})

test("native preview uses the mobile-safe worker without changing a muted subscription", async () => {
  localStorage.setItem("supportchat.push", "muted-device")
  const shown: { title: string; options: NotificationOptions }[] = []
  vi.stubGlobal(
    "Notification",
    class {
      static permission = "granted"
      constructor() {
        throw new Error("Mobile browsers require a service worker")
      }
    },
  )
  vi.stubGlobal("PushManager", undefined)
  Object.defineProperty(navigator, "serviceWorker", {
    configurable: true,
    value: {
      getRegistration: async () => ({
        active: {},
        showNotification: async (title: string, options: NotificationOptions) => {
          shown.push({ title, options })
        },
      }),
    },
  })
  expect(await previewSystemSound()).toBe(true)
  expect(await previewSystemSound()).toBe(true)
  expect(shown).toEqual([
    { title: "SupportChat sound test", options: expect.objectContaining({ silent: false }) },
    { title: "SupportChat sound test", options: expect.objectContaining({ silent: false }) },
  ])
  expect(new Set(shown.map((item) => item.options.tag)).size).toBe(2)
  expect(localStorage.getItem("supportchat.push")).toBe("muted-device")
})

test("blocked permission cannot claim that a native sound preview was sent", async () => {
  vi.stubGlobal("Notification", { permission: "denied" })
  expect(await previewSystemSound()).toBe(false)
})

test("permission denial never creates a subscription", async () => {
  vi.stubGlobal("Notification", { permission: "denied", requestPermission: async () => "denied" })
  await expect(enablePush(false)).rejects.toThrow("blocked")
  expect(hasActivePush()).toBe(false)
})

test("failed server registration unsubscribes the new browser device", async () => {
  const device = {
    endpoint: "https://fcm.googleapis.com/fcm/send/device-a",
    toJSON: () => ({
      endpoint: "https://fcm.googleapis.com/fcm/send/device-a",
      keys: { p256dh: "public", auth: "auth" },
    }),
    unsubscribe: vi.fn(async () => true),
  }
  vi.stubGlobal("Notification", { permission: "granted" })
  vi.stubGlobal("PushManager", function PushManager() {
    return null
  })
  Object.defineProperty(navigator, "serviceWorker", {
    configurable: true,
    value: {
      register: async () => ({
        active: { postMessage: vi.fn() },
        pushManager: { getSubscription: async () => null, subscribe: async () => device },
      }),
      ready: Promise.resolve({}),
    },
  })
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async (url) =>
        new Response(
          JSON.stringify(
            String(url).endsWith("config") ? { configured: true, public_key: "AQID" } : {},
          ),
          { status: String(url).endsWith("config") ? 200 : 503 },
        ),
    ),
  )
  await expect(enablePush(false)).rejects.toThrow("save")
  expect(device.unsubscribe).toHaveBeenCalledOnce()
  expect(hasActivePush()).toBe(false)
})

test("disable clears local delivery even when the server is unavailable", async () => {
  localStorage.setItem("supportchat.push", "device-a")
  const unsubscribe = vi.fn(async () => true)
  const messages: unknown[] = []
  Object.defineProperty(navigator, "serviceWorker", {
    configurable: true,
    value: {
      getRegistration: async () => ({
        active: {
          postMessage: (message: unknown, ports: MessagePort[]) => {
            messages.push(message)
            ports[0].postMessage({ ok: true }, [])
          },
        },
        getNotifications: async () => [],
        pushManager: {
          getSubscription: async () => ({
            endpoint: "https://fcm.googleapis.com/fcm/send/device-a",
            unsubscribe,
          }),
        },
      }),
    },
  })
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      throw new Error("offline")
    }),
  )
  await disablePush()
  expect(hasActivePush()).toBe(false)
  expect(messages).toContainEqual({ type: "supportchat.push.configure", config: null })
  expect(unsubscribe).toHaveBeenCalledOnce()
})

test("a delayed worker acknowledgement cannot restore push after logout", async () => {
  const entered = Promise.withResolvers<void>()
  let acknowledge!: () => void
  vi.stubGlobal("Notification", { permission: "granted" })
  vi.stubGlobal("PushManager", function PushManager() {
    return null
  })
  const device = {
    endpoint: "https://fcm.googleapis.com/fcm/send/device-a",
    toJSON: () => ({ keys: { p256dh: "public", auth: "auth" } }),
    unsubscribe: async () => true,
  }
  const registration = {
    active: {
      postMessage: (message: { config: unknown }, ports: MessagePort[]) => {
        if (message.config === null) ports[0].postMessage({ ok: true }, [])
        else {
          acknowledge = () => ports[0].postMessage({ ok: true }, [])
          entered.resolve()
        }
      },
    },
    getNotifications: async () => [],
    pushManager: { getSubscription: async () => device },
  }
  Object.defineProperty(navigator, "serviceWorker", {
    configurable: true,
    value: { getRegistration: async () => registration },
  })
  vi.stubGlobal("fetch", async () => Response.json({ id: "22222222-2222-4222-8222-000000000002" }))
  const syncing = syncExistingPush(false)
  await entered.promise
  await disablePush()
  acknowledge()
  await expect(syncing).rejects.toThrow("cancelled")
  expect(hasActivePush()).toBe(false)
})

const DEVICE_ID = "22222222-2222-4222-8222-000000000002"
const SERVER_KEY = "AQID"

const pushHarness = (existing: object | null) => {
  const posts: string[] = []
  const subscribe = vi.fn(async () => ({
    endpoint: "https://fcm.googleapis.com/fcm/send/new",
    options: { applicationServerKey: new Uint8Array([1, 2, 3]).buffer },
    toJSON: () => ({ keys: { p256dh: "public", auth: "auth" } }),
    unsubscribe: vi.fn(async () => true),
  }))
  vi.stubGlobal("Notification", { permission: "granted" })
  vi.stubGlobal("PushManager", function PushManager() {
    return null
  })
  const registration = {
    active: {
      postMessage: (_message: unknown, ports: MessagePort[]) =>
        ports[0].postMessage({ ok: true }, []),
    },
    pushManager: { getSubscription: async () => existing, subscribe },
  }
  Object.defineProperty(navigator, "serviceWorker", {
    configurable: true,
    value: { getRegistration: async () => registration, register: async () => registration },
  })
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      if (String(url).endsWith("config"))
        return Response.json({ configured: true, public_key: SERVER_KEY })
      if (init?.method === "POST") posts.push(String(url))
      return Response.json({ id: DEVICE_ID })
    }),
  )
  return { subscribe, posts }
}

test("a subscription made with a rotated VAPID key is replaced on load", async () => {
  const unsubscribe = vi.fn(async () => true)
  const { subscribe, posts } = pushHarness({
    endpoint: "https://fcm.googleapis.com/fcm/send/old",
    options: { applicationServerKey: new Uint8Array([9, 9, 9]).buffer },
    toJSON: () => ({ keys: { p256dh: "public", auth: "auth" } }),
    unsubscribe,
  })
  expect(await syncExistingPush(false)).toBe(true)
  expect(unsubscribe).toHaveBeenCalledOnce()
  expect(subscribe).toHaveBeenCalledOnce()
  expect(posts).toHaveLength(1)
  expect(hasActivePush()).toBe(true)
})

test("a subscription the browser dropped is recreated for a device that enabled push", async () => {
  localStorage.setItem("supportchat.push", DEVICE_ID)
  const { subscribe, posts } = pushHarness(null)
  expect(await syncExistingPush(false)).toBe(true)
  expect(subscribe).toHaveBeenCalledOnce()
  expect(posts).toHaveLength(1)
})

test("a device that never enabled push is not subscribed on load", async () => {
  const { subscribe } = pushHarness(null)
  expect(await syncExistingPush(false)).toBe(false)
  expect(subscribe).not.toHaveBeenCalled()
})

test("revoked browser permission turns push off locally", async () => {
  localStorage.setItem("supportchat.push", DEVICE_ID)
  const { subscribe, posts } = pushHarness(null)
  vi.stubGlobal("Notification", { permission: "denied" })
  expect(await syncExistingPush(false)).toBe(false)
  expect(hasActivePush()).toBe(false)
  expect(subscribe).not.toHaveBeenCalled()
  expect(posts).toHaveLength(0)
})
