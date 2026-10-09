"use client"

import { useCallback, useState, type ChangeEvent, type KeyboardEvent } from "react"

import { Button } from "@/components/ui/button"
import { Field, FieldDescription, FieldError, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

import {
  addTopic,
  ASSISTANT_SCOPE_MAX,
  HANDOFF_TOPIC_MAX,
  HANDOFF_TOPICS_MAX,
  type AssistantProfile,
} from "./assistant-profile"
import { FIELD } from "./sites-shared"

const TopicRow = ({
  topic,
  disabled,
  onRemove,
}: {
  topic: string
  disabled: boolean
  onRemove: (topic: string) => void
}) => {
  const handleClick = useCallback(() => onRemove(topic), [onRemove, topic])
  return (
    <li className="border-line bg-ice flex items-center justify-between gap-2 rounded-lg border px-3 py-1.5 text-sm">
      <span className="min-w-0 break-words">{topic}</span>
      {disabled ? null : (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-label={`Remove ${topic}`}
          onClick={handleClick}
          className="text-steel shrink-0 cursor-pointer font-bold"
        >
          Remove
        </Button>
      )}
    </li>
  )
}

const TopicAdder = ({
  id,
  topics,
  onChange,
}: {
  id: string
  topics: string[]
  onChange: (topics: string[]) => void
}) => {
  const [draft, setDraft] = useState("")
  const full = topics.length >= HANDOFF_TOPICS_MAX
  const commit = useCallback(() => {
    onChange(addTopic(topics, draft))
    setDraft("")
  }, [draft, onChange, topics])
  const handleDraft = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setDraft(event.target.value)
  }, [])
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key !== "Enter") return
      event.preventDefault()
      commit()
    },
    [commit],
  )
  return (
    <div className="flex flex-col gap-2 sm:flex-row">
      <Input
        id={id}
        name="handoffTopic"
        autoComplete="off"
        value={draft}
        maxLength={HANDOFF_TOPIC_MAX}
        disabled={full}
        placeholder={full ? "Topic limit reached" : "Describe a topic, then press Enter"}
        onChange={handleDraft}
        onKeyDown={handleKeyDown}
        className={FIELD}
      />
      <Button
        type="button"
        variant="outline"
        onClick={commit}
        disabled={full || draft.trim().length === 0}
        className="cursor-pointer font-bold"
      >
        Add topic
      </Button>
    </div>
  )
}

const HandoffTopics = ({
  siteId,
  topics,
  disabled,
  onChange,
}: {
  siteId: string
  topics: string[]
  disabled: boolean
  onChange: (topics: string[]) => void
}) => {
  const handleRemove = useCallback(
    (topic: string) => onChange(topics.filter((item) => item !== topic)),
    [onChange, topics],
  )
  const inputId = `manage-handoff-${siteId}`
  return (
    <Field>
      <FieldLabel htmlFor={inputId}>Always hand off to a person for</FieldLabel>
      <FieldDescription>
        Topics the assistant never answers itself. Visitors asking about them are passed to your
        team.
      </FieldDescription>
      {topics.length > 0 ? (
        <ul className="flex flex-col gap-1.5" aria-label="Hand-off topics">
          {topics.map((topic) => (
            <TopicRow key={topic} topic={topic} disabled={disabled} onRemove={handleRemove} />
          ))}
        </ul>
      ) : (
        <p className="text-mute text-xs">No hand-off topics yet.</p>
      )}
      {disabled ? null : <TopicAdder id={inputId} topics={topics} onChange={onChange} />}
    </Field>
  )
}

const LanguageField = ({
  siteId,
  value,
  disabled,
  problem,
  onChange,
  onBlur,
}: {
  siteId: string
  value: string
  disabled: boolean
  problem?: string
  onChange: (event: ChangeEvent<HTMLInputElement>) => void
  onBlur: () => void
}) => {
  const errorId = `manage-language-${siteId}-error`
  return (
    <Field>
      <FieldLabel htmlFor={`manage-language-${siteId}`}>Language staff write in</FieldLabel>
      <Input
        id={`manage-language-${siteId}`}
        name="primaryLanguage"
        autoComplete="off"
        value={value}
        disabled={disabled}
        onChange={onChange}
        onBlur={onBlur}
        aria-invalid={problem ? "true" : undefined}
        aria-describedby={errorId}
        className={`${FIELD} sm:max-w-48`}
      />
      <FieldDescription>
        The language canned replies and knowledge are written in, as a code such as en, es, or
        pt-BR.
      </FieldDescription>
      <FieldError id={errorId}>{problem}</FieldError>
    </Field>
  )
}

export const ManageSiteAssistant = ({
  siteId,
  isAdmin,
  profile,
  languageProblem,
  onProfile,
  onLanguageBlur,
}: {
  siteId: string
  isAdmin: boolean
  profile: AssistantProfile
  languageProblem?: string
  onProfile: (next: AssistantProfile) => void
  onLanguageBlur: () => void
}) => {
  const handleScope = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) =>
      onProfile({ ...profile, scope: event.target.value }),
    [onProfile, profile],
  )
  const handleTopics = useCallback(
    (topics: string[]) => onProfile({ ...profile, topics }),
    [onProfile, profile],
  )
  const handleLanguage = useCallback(
    (event: ChangeEvent<HTMLInputElement>) =>
      onProfile({ ...profile, language: event.target.value }),
    [onProfile, profile],
  )
  return (
    <section className="border-line flex flex-col gap-4 border-t pt-5" aria-label="Assistant">
      <div>
        <p className="text-mute text-[10px] font-semibold tracking-[0.12em] uppercase">Assistant</p>
        <h3 className="text-navy heading text-sm">What this assistant knows and does</h3>
      </div>
      <Field>
        <FieldLabel htmlFor={`manage-scope-${siteId}`}>What this assistant covers</FieldLabel>
        <Textarea
          id={`manage-scope-${siteId}`}
          name="assistantScope"
          autoComplete="off"
          value={profile.scope}
          maxLength={ASSISTANT_SCOPE_MAX}
          disabled={!isAdmin}
          onChange={handleScope}
          className={`${FIELD} leading-6`}
          rows={4}
        />
        <FieldDescription>
          Describe the subjects visitors can ask about. Leave empty to infer this from the knowledge
          base. {profile.scope.length}/{ASSISTANT_SCOPE_MAX}
        </FieldDescription>
      </Field>
      <HandoffTopics
        siteId={siteId}
        topics={profile.topics}
        disabled={!isAdmin}
        onChange={handleTopics}
      />
      <LanguageField
        siteId={siteId}
        value={profile.language}
        disabled={!isAdmin}
        problem={languageProblem}
        onChange={handleLanguage}
        onBlur={onLanguageBlur}
      />
    </section>
  )
}
