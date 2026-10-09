#!/usr/bin/env bash
# Fail unless the canonical CI/CD workflow succeeded for a commit (manual deploy gate).
#
# Usage: verify-ci-checks.sh <sha> <owner/repo>
# Requires: gh CLI authenticated (GH_TOKEN / GITHUB_TOKEN in CI).
#
# Override via env:
#   CANONICAL_WORKFLOW (default .github/workflows/ci-cd.yml)
#   REQUIRED_JOBS      (newline-separated job names)
#   CHANGES_JOB        (path-filter job name; must be success if present)
# Required jobs may be "skipped" (path filtering) but not failed/cancelled/missing.
set -euo pipefail

SHA="${1:?commit sha required}"
REPO="${2:?owner/repo required}"
CANONICAL_WORKFLOW="${CANONICAL_WORKFLOW:-.github/workflows/ci-cd.yml}"

if [ -n "${REQUIRED_JOBS:-}" ]; then
  mapfile -t required_jobs <<< "${REQUIRED_JOBS}"
else
  required_jobs=(
    "Backend (lint + test)"
    "Frontend (lint + typecheck + test + build)"
  )
fi

mapfile -t runs < <(
  gh api "repos/${REPO}/actions/runs?head_sha=${SHA}" \
    --paginate \
    -q ".workflow_runs[] | select(.path == \"${CANONICAL_WORKFLOW}\") | \"\(.id)\t\(.status)\t\(.conclusion // \"\")\"" \
    2>/dev/null || true
)

if [ "${#runs[@]}" -eq 0 ]; then
  echo "Canonical workflow ${CANONICAL_WORKFLOW} has no runs for ${SHA}"
  exit 1
fi

fail=0
run_ids=()
for row in "${runs[@]}"; do
  IFS=$'\t' read -r run_id status conclusion <<< "${row}"
  run_ids+=("${run_id}")
  if [ "${status}" != "completed" ]; then
    echo "Canonical workflow run ${run_id} not completed (status=${status})"
    fail=1
    continue
  fi
  if [ "${conclusion}" != "success" ]; then
    echo "Canonical workflow run ${run_id} conclusion: ${conclusion}"
    fail=1
  fi
done

for name in "${required_jobs[@]}"; do
  [ -n "$name" ] || continue
  mapfile -t jobs < <(
    for run_id in "${run_ids[@]}"; do
      gh api "repos/${REPO}/actions/runs/${run_id}/jobs" \
        --paginate \
        -q ".jobs[] | select(.name == \"${name}\") | \"\(.status)\t\(.conclusion // \"\")\"" \
        2>/dev/null || true
    done
  )
  if [ "${#jobs[@]}" -eq 0 ]; then
    echo "Required job '${name}' not found on ${CANONICAL_WORKFLOW} for ${SHA}"
    fail=1
    continue
  fi
  for job_row in "${jobs[@]}"; do
    IFS=$'\t' read -r status conclusion <<< "${job_row}"
    if [ "${status}" != "completed" ]; then
      echo "Required job '${name}' not completed (status=${status})"
      fail=1
      continue
    fi
    # "skipped" = job-level path filter said this side did not change.
    if [ "${conclusion}" != "success" ] && [ "${conclusion}" != "skipped" ]; then
      echo "Required job '${name}' conclusion: ${conclusion}"
      fail=1
    fi
  done
done

# The path-filter job decides which required jobs may be skipped. If it is
# present it must have succeeded, otherwise a broken filter could skip every
# job and let an untested commit through. (Absent on runs from older workflows.)
CHANGES_JOB="${CHANGES_JOB:-Detect changed paths}"
for run_id in "${run_ids[@]}"; do
  mapfile -t change_rows < <(
    gh api "repos/${REPO}/actions/runs/${run_id}/jobs" \
      --paginate \
      -q ".jobs[] | select(.name == \"${CHANGES_JOB}\") | \"\(.status)\t\(.conclusion // \"\")\"" \
      2>/dev/null || true
  )
  for job_row in "${change_rows[@]}"; do
    IFS=$'\t' read -r status conclusion <<< "${job_row}"
    if [ "${status}" != "completed" ] || [ "${conclusion}" != "success" ]; then
      echo "Job '${CHANGES_JOB}' did not succeed (status=${status}, conclusion=${conclusion})"
      fail=1
    fi
  done
done

if [ "$fail" -ne 0 ]; then
  echo "CI verification failed for ${SHA} — deploy blocked."
  exit 1
fi

echo "All required CI checks passed for ${SHA}."
