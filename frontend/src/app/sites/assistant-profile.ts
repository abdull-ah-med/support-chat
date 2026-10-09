export const ASSISTANT_SCOPE_MAX = 2000
export const HANDOFF_TOPICS_MAX = 20
export const HANDOFF_TOPIC_MAX = 200
export const DEFAULT_LANGUAGE = "en"

const LANGUAGE_PATTERN = /^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$/

export type AssistantProfile = {
  scope: string
  topics: string[]
  language: string
}

export const profileOf = (site: {
  assistant_scope?: string
  handoff_topics?: string[]
  primary_language?: string
}): AssistantProfile => ({
  scope: site.assistant_scope ?? "",
  topics: site.handoff_topics ?? [],
  language: site.primary_language ?? DEFAULT_LANGUAGE,
})

export const languageError = (value: string) =>
  LANGUAGE_PATTERN.test(value.trim()) ? null : "Use a language code such as en, es, or pt-BR."

export const isProfileDirty = (site: Parameters<typeof profileOf>[0], next: AssistantProfile) => {
  const saved = profileOf(site)
  return (
    next.scope !== saved.scope ||
    next.language !== saved.language ||
    next.topics.length !== saved.topics.length ||
    next.topics.some((topic, index) => topic !== saved.topics[index])
  )
}

/** Returns the list with `raw` added, or the same list when it is blank, a repeat, or over a limit. */
export const addTopic = (topics: string[], raw: string) => {
  const topic = raw.replace(/\s+/g, " ").trim()
  if (!topic || topic.length > HANDOFF_TOPIC_MAX || topics.length >= HANDOFF_TOPICS_MAX) {
    return topics
  }
  const key = topic.toLowerCase()
  if (topics.some((existing) => existing.toLowerCase() === key)) return topics
  return [...topics, topic]
}

export const assistantPayload = (profile: AssistantProfile) => ({
  assistant_scope: profile.scope.trim(),
  handoff_topics: profile.topics,
  primary_language: profile.language.trim(),
})
