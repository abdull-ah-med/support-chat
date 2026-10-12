# Support Chat

An embeddable live-chat platform with a grounded support bot and a staff inbox.

Visitors chat with a widget that is dropped into any website with a single script tag. A bot answers from a per-site knowledge base and cites its sources. When a conversation needs a person, staff pick it up from an admin inbox and the bot steps aside.

## Stack

| Layer | Technology |
|-------|------------|
| API and WebSockets | FastAPI on Python 3.12, managed with [uv](https://docs.astral.sh/uv/) |
| Staff app and widget | Next.js 16, React 19, TypeScript |
| Database | PostgreSQL 16 with pgvector, migrated with Alembic |
| Cache, pub/sub, rate limits | Redis 7 |
| Bot generation | Anthropic API |
| Embeddings | OpenAI API |
| Packaging | Docker Compose |

## Repository layout

```
backend/                     FastAPI app, WebSocket handlers, workers, Alembic migrations, tests
frontend/                    Next.js staff app, widget iframe surface, embed loader
deploy/                      Host bootstrap, nginx template, deploy and rollback scripts
docker-compose.yml           Local full stack
docker-compose.prod.yml      Production overlay (TLS Redis, split API and worker)
.github/workflows/           CI and deploy pipelines
```

## How it works

### Surfaces

The backend separates three origins by host:

- **Staff app** (`STAFF_APP_ORIGIN`): inbox, sites, knowledge base, logs, settings. JWT auth with rotating refresh cookies and CSRF protection on mutating requests.
- **Widget** (`WIDGET_ORIGIN`): the visitor chat UI, loaded in a cross-origin iframe. Callers are checked against each site's origin allowlist.
- **Demo host** (`MARKETING_HOST_ORIGIN`): a page that loads the embed loader, used for local testing.

### Widget embed

A host page loads `supportchat.js`, built from `frontend/embed-loader/`. The loader renders a launcher and opens the widget iframe. Visitor identity is scoped to the site key, so it does not carry across sites, and host page CSS cannot reach the transcript.

```bash
cd frontend && npm run build:loader
```

### Chat runtime

- PostgreSQL stores conversations, messages, visitors, handoffs and knowledge data.
- WebSockets carry live transcript and state to visitors and agents.
- Redis pub/sub fans events out across API processes.
- When an agent joins a conversation the bot stops generating for it.
- Background workers (a separate Compose service in production) handle knowledge ingest, idle-close, maintenance purges, location enrichment and push notifications.

### Knowledge base

Sources are crawled or fetched, extracted, chunked, embedded and indexed in PostgreSQL with pgvector. The bot retrieves evidence with hybrid search, validates its draft against the retrieved sources, and hands off to a person when it cannot answer. Crawl and fetch paths block SSRF targets such as private IP ranges, metadata endpoints and non-HTTP schemes.

## Getting started

### Prerequisites

- Docker Engine with Compose v2
- Node.js 22 and npm
- uv 0.11.x
- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` if you want bot replies and knowledge ingest

### 1. Environment files

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```

Set unique values of at least 64 characters for `JWT_SECRET`, `WIDGET_TOKEN_SECRET`, `RATE_KEY_SECRET` and `WIDGET_CSP_SERVICE_SECRET`. Never commit real secrets.

### 2. Full stack with Docker

```bash
cd backend && ./scripts/docker_up.sh
```

This builds and starts Postgres (`55432`), Redis (`56379`), the API (`8000`), a worker and the frontend (`3000`). The API container runs migrations and seeds a staff user on first boot.

### 3. Host-run API and frontend (optional)

With the Compose Postgres and Redis running:

```bash
cd backend
uv sync --frozen
uv run alembic upgrade head
uv run python -m scripts.seed_user      # once
uv run fastapi dev --host 0.0.0.0 --port 8000
```

```bash
cd frontend
npm ci
npm run dev
```

Point `DATABASE_URL` and `REDIS_URL` in `backend/.env` at the Compose ports.

### 4. Local hostnames

| Variable | Default |
|----------|---------|
| `STAFF_APP_ORIGIN` | `http://localhost:3000` |
| `WIDGET_ORIGIN` | `http://widget.localhost:3000` |
| `MARKETING_HOST_ORIGIN` | `http://host.localhost:3000` |

Most browsers resolve `*.localhost` automatically. Otherwise add the hosts to `/etc/hosts`.

## Development

### Backend

The test database name must contain `test` (default `support_chat_test`).

```bash
cd backend
uv run ruff check .
uv run ruff format --check .
CI=true uv run pytest
```

### Frontend

```bash
cd frontend
npm run lint
npm run typecheck
npm test
npm run build
```

### Database migrations

```bash
cd backend
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "describe change"
```

Migrations live in `backend/alembic/versions/`. Schema changes go through Alembic only.

## Configuration

See `.env.example` and `backend/.env.example` for the full list. The main settings:

| Variable | Purpose |
|----------|---------|
| `APP_ENV` | `local`, `staging` or `production`. Production enforces secure cookies and required secrets. |
| `ENABLE_BACKGROUND_WORKERS` | `false` on API containers, `true` on the worker service. |
| `CHAT_RETENTION_DAYS` | How long conversations are kept. |
| `TRUSTED_PROXY_CIDRS` | Proxies allowed to set forwarded client IP headers. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Required outside `APP_ENV=test` for the bot and search. |

Frontend build-time URLs use the `NEXT_PUBLIC_*` variables.

## Deployment

`.github/workflows/ci-cd.yml` runs backend and frontend checks, builds images, and deploys to a host through the scripts in `deploy/`. The deploy flow includes a preflight check, a database backup before migration, a smoke test and automatic rollback. See `deploy/ROLLBACK.md` for manual recovery.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Production runs the API and worker as separate containers, and Redis with TLS (`deploy/generate-redis-tls.sh`). To reproduce the CI checks locally:

```bash
bash deploy/verify-ci-checks.sh
```

## Operational scripts

| Script | Purpose |
|--------|---------|
| `backend/scripts/seed_user.py` | Create the initial staff user |
| `backend/scripts/generate_push_keys.py` | Generate web push VAPID keys |
| `backend/scripts/purge_expired_chats.py` | Enforce chat retention |
| `backend/scripts/purge_expired_logs.py` | Trim old application logs |
| `backend/scripts/purge_expired_refresh_tokens.py` | Remove expired refresh tokens |
| `backend/scripts/run_site_eval.py` | Offline bot evaluation against golden sets |

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Latest integrated work. Same tip as `dev`. |
| `dev` | Day-to-day integration branch. |
| `staging` | Earlier pre-release snapshot. |

`staging` predates a large amount of later work, so some sections above may describe functionality it does not have yet.

## Security notes

- Transcripts can contain personal information. Do not log message bodies, prompts that include visitor text, or exports in unsecured channels.
- Widget and admin surfaces are isolated by host and site key. Visitor identity is never shared between sites.
- Rate limits cover bootstrap, visitor creation, message submit, login failures and widget CSP endpoints.
