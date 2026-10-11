# Multi-stage production image using Next.js `output: "standalone"`.
# https://github.com/vercel/next.js/tree/canary/examples/with-docker
# https://nextjs.org/docs/app/api-reference/config/next-config-js/output

# TODO(INF-11): pin by digest once a build against it has been verified. Index digest seen
# 2026-10-10: node:22-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392
FROM node:22-bookworm-slim AS base
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

FROM base AS deps
COPY package.json package-lock.json ./
# Skip husky prepare (needs git hooks outside the image).
RUN npm ci --ignore-scripts

FROM base AS build
ARG API_ORIGIN=http://backend:8000
ARG NEXT_PUBLIC_API_ORIGIN=
ARG NEXT_PUBLIC_WIDGET_ORIGIN=
ARG NEXT_PUBLIC_STAFF_APP_ORIGIN=
ARG NEXT_PUBLIC_MARKETING_HOST_ORIGIN=
ARG NEXT_PUBLIC_DEMO_SITE_KEY=demo
ARG NEXT_PUBLIC_DEMO_PUBLIC_KEY=

ENV API_ORIGIN=$API_ORIGIN \
    NEXT_PUBLIC_API_ORIGIN=$NEXT_PUBLIC_API_ORIGIN \
    NEXT_PUBLIC_WIDGET_ORIGIN=$NEXT_PUBLIC_WIDGET_ORIGIN \
    NEXT_PUBLIC_STAFF_APP_ORIGIN=$NEXT_PUBLIC_STAFF_APP_ORIGIN \
    NEXT_PUBLIC_MARKETING_HOST_ORIGIN=$NEXT_PUBLIC_MARKETING_HOST_ORIGIN \
    NEXT_PUBLIC_DEMO_SITE_KEY=$NEXT_PUBLIC_DEMO_SITE_KEY \
    NEXT_PUBLIC_DEMO_PUBLIC_KEY=$NEXT_PUBLIC_DEMO_PUBLIC_KEY \
    NODE_ENV=production

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM base AS production
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=build --chown=appuser:appuser /app/public ./public
RUN mkdir .next && chown appuser:appuser .next

COPY --from=build --chown=appuser:appuser /app/.next/standalone ./
COPY --from=build --chown=appuser:appuser /app/.next/static ./.next/static

USER appuser
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=12 \
    CMD ["node", "-e", "fetch('http://127.0.0.1:3000/login').then((r)=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"]
CMD ["node", "server.js"]
