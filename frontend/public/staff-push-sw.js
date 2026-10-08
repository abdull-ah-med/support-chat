/* global self, caches */
// No fetch handler: this worker never caches pages, API responses, or chat content.
const DEVICE_CACHE = "supportchat-push-device"
const CONFIG_URL = new URL("/__supportchat_push_device", self.location.origin).href
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

const staffWindow = (client) => {
  const url = new URL(client.url)
  return url.origin === self.location.origin && /^\/(admin|inbox|settings)(\/|$)/.test(url.pathname)
}

const saveDevice = async (config) => {
  const cache = await caches.open(DEVICE_CACHE)
  if (config === null) await cache.delete(CONFIG_URL)
  else await cache.put(CONFIG_URL, new Response(JSON.stringify(config)))
}

const readDevice = async () => {
  const cache = await caches.open(DEVICE_CACHE)
  const response = await cache.match(CONFIG_URL)
  return response ? response.json() : null
}

self.addEventListener("install", (event) => event.waitUntil(self.skipWaiting()))
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()))
const validConfig = (config) =>
  config === null || (UUID.test(config?.subscription_id) && typeof config?.silent === "boolean")

let configurationWrite = Promise.resolve()
self.addEventListener("message", (event) => {
  if (
    !event.source?.url ||
    !staffWindow(event.source) ||
    event.data?.type !== "supportchat.push.configure"
  )
    return
  const config = event.data.config
  if (!validConfig(config)) return
  configurationWrite = configurationWrite.catch(() => undefined).then(() => saveDevice(config))
  event.waitUntil(
    configurationWrite.then(
      () => event.ports[0]?.postMessage({ ok: true }, []),
      () => event.ports[0]?.postMessage({ ok: false }, []),
    ),
  )
})

const validPayload = (payload) => {
  if (typeof payload.title !== "string" || payload.title.length > 200) return false
  if (typeof payload.body !== "string" || payload.body.length > 1000) return false
  const id = payload.notification_id
  if (!(Number.isSafeInteger(id) && id > 0) && !/^test-\d+$/.test(id)) return false
  return payload.conversation_id === null || UUID.test(payload.conversation_id)
}

const parsePayload = (event) => {
  try {
    return event.data.json()
  } catch {
    return null
  }
}

const signalTestSound = (windows, isTest, silent) => {
  if (!isTest || silent) return
  // Signal actual delivery, once, to one tab. It can supply a tone where the
  // browser's native implementation has no sound support.
  const client = windows.find((window) => window.focused) ?? windows[0]
  client?.postMessage({ type: "supportchat.push.test" }, [])
}

const FALLBACK = {
  title: "SupportChat · New activity",
  body: "Open SupportChat to view your inbox.",
  notification_id: null,
  conversation_id: null,
}

const resolvePayload = async (event) => {
  const parsed = parsePayload(event)
  const device = await readDevice().catch(() => null)
  const payload = parsed && validPayload(parsed) ? parsed : FALLBACK
  return { device, payload }
}

const staffWindows = async () =>
  (await self.clients.matchAll({ type: "window", includeUncontrolled: true })).filter(staffWindow)

const notificationTag = (payload, device) =>
  payload === FALLBACK
    ? "supportchat:activity"
    : `supportchat:${payload.subscription_id ?? device?.subscription_id}:${payload.notification_id}`

const inboxUrl = (payload) => {
  const url = new URL("/admin/inbox", self.location.origin)
  if (payload.conversation_id) url.searchParams.set("conversation", payload.conversation_id)
  return url.href
}

const receivePush = async (event) => {
  // A push event must always end in a visible notification: browsers penalise silent pushes
  // (Chrome shows a generic notice, Safari revokes the subscription). The payload is encrypted
  // to this browser's own subscription, so it needs no further device matching; the stored
  // device configuration only supplies the mute preference.
  const { device, payload } = await resolvePayload(event)
  const windows = await staffWindows()
  for (const client of windows) client.postMessage({ type: "supportchat.push" }, [])
  const isTest = payload.conversation_id === null && payload !== FALLBACK
  const tag = notificationTag(payload, device)
  // Retries after a worker restart cannot produce a second alert while this one is displayed.
  if ((await self.registration.getNotifications({ tag })).length > 0) return
  const silent = (device?.silent ?? false) || payload.silent === true
  await self.registration.showNotification(payload.title, {
    body: payload.body,
    tag,
    renotify: payload === FALLBACK,
    icon: "/icon-watermark.png",
    silent,
    actions: [{ action: "open_inbox", title: isTest ? "Open inbox" : "View chat" }],
    data: { url: inboxUrl(payload) },
  })
  signalTestSound(windows, isTest, silent)
}
let incomingPush = Promise.resolve()
self.addEventListener("push", (event) => {
  incomingPush = incomingPush.catch(() => undefined).then(() => receivePush(event))
  event.waitUntil(incomingPush)
})

const openChat = async (event) => {
  event.notification.close()
  let url
  try {
    url = new URL(event.notification.data?.url)
  } catch {
    return
  }
  if (url.origin !== self.location.origin || url.pathname !== "/admin/inbox") return
  const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true })
  const existing = windows.find(staffWindow)
  if (existing) {
    const navigated = await existing.navigate(url.href)
    if (navigated) {
      await navigated.focus()
      return
    }
  }
  await self.clients.openWindow(url.href)
}
self.addEventListener("notificationclick", (event) => event.waitUntil(openChat(event)))
const resubscribe = async (event) => {
  // The browser rotated or dropped the subscription. Recreate it with the same server key so the
  // device keeps receiving pushes; open windows re-register the new endpoint with the backend.
  // Without an open window the next visit does it (the old endpoint is pruned on 410).
  try {
    const key = event.oldSubscription?.options?.applicationServerKey
    if (!event.newSubscription && key) {
      await self.registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: key,
      })
    }
  } catch {
    /* A window will resubscribe on the next visit. */
  }
  const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true })
  for (const client of windows.filter(staffWindow))
    client.postMessage({ type: "supportchat.push.expired" }, [])
}
self.addEventListener("pushsubscriptionchange", (event) => event.waitUntil(resubscribe(event)))
