#!/usr/bin/env python3
"""Read resolved Compose values without logging credentials. No third-party modules."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


def compose_output(*args):
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            os.getenv("COMPOSE_FILE", "docker-compose.yml"),
            "-f",
            os.getenv("COMPOSE_PROD_FILE", "docker-compose.prod.yml"),
            "config",
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        # Compose diagnostics can contain interpolated secrets. Print variable names only.
        names = re.findall(
            r"(?:required variable|variable) [\"']?([A-Z][A-Z0-9_]*)", result.stderr
        )
        raise ValueError(
            "Compose config failed; check root .env / backend env file and required variables"
            + (": " + ", ".join(sorted(set(names))) if names else "")
        )
    return result.stdout


def memory_bytes(value):
    match = re.fullmatch(r"(\d+)([kmgt]?)(?:i?b)?", str(value).lower())
    if not match:
        raise ValueError("Invalid Redis memory limit")
    return int(match[1]) * 1024 ** (" kmgt".index(match[2]) if match[2] else 0)


def main():
    mode = sys.argv[1]
    config = json.loads(compose_output("--format", "json"))
    services = config["services"]
    if mode == "validate":
        if not Path(".env").is_file():
            raise ValueError("Missing root .env")
        env = dict(
            line.split("=", 1)
            for line in compose_output("--environment").splitlines()
            if "=" in line
        )
        # Discover all required substitutions, including future additions to either overlay.
        text = "\n".join(
            Path(os.getenv(key, default)).read_text()
            for key, default in (
                ("COMPOSE_FILE", "docker-compose.yml"),
                ("COMPOSE_PROD_FILE", "docker-compose.prod.yml"),
            )
        )
        text = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        required = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):\?", text))
        missing = sorted(key for key in required if not env.get(key))
        if missing:
            raise ValueError("Missing root .env variables: " + ", ".join(missing))
        if env.get("APP_ENV") != "production":
            raise ValueError("Root .env must set APP_ENV=production")
        # HEAD settings are also read by healthchecks/exec, which bypass the entrypoint.
        backend_env = Path(env.get("BACKEND_ENV_FILE") or "backend/.env.prod")
        match = re.search(
            r"(?m)^TRUSTED_PROXY_CIDRS=['\"]?([0-9.]+/32)['\"]?\s*(?:#.*)?$",
            backend_env.read_text(),
        )
        import ipaddress

        if not match:
            raise ValueError(
                "backend env must set TRUSTED_PROXY_CIDRS to a single gateway /32; inspect the Docker network"
            )
        ipaddress.IPv4Network(match[1])
    elif mode == "origins":
        args = services["frontend"]["build"]["args"]
        hosts = []
        for key in (
            "NEXT_PUBLIC_STAFF_APP_ORIGIN",
            "NEXT_PUBLIC_WIDGET_ORIGIN",
            "NEXT_PUBLIC_MARKETING_HOST_ORIGIN",
        ):
            origin = urlsplit(args[key])
            if (
                origin.scheme != "https"
                or not origin.hostname
                or origin.port not in (None, 443)
                or origin.path not in ("", "/")
                or origin.query
                or origin.fragment
                or origin.username
                or origin.password
                or not re.fullmatch(r"[a-zA-Z0-9.-]+", origin.hostname)
            ):
                raise ValueError("Invalid public HTTPS origin: " + key)
            hosts.append(origin.hostname)
        if len(set(hosts)) != 3:
            raise ValueError("Public origins must be distinct")
        print("\n".join(hosts))
    elif mode == "build-args":
        # Same keys CI bakes into the frontend image (see package-images.sh manifest).
        args = services["frontend"]["build"]["args"]
        keys = (
            "API_ORIGIN",
            "NEXT_PUBLIC_API_ORIGIN",
            "NEXT_PUBLIC_WIDGET_ORIGIN",
            "NEXT_PUBLIC_STAFF_APP_ORIGIN",
            "NEXT_PUBLIC_MARKETING_HOST_ORIGIN",
            "NEXT_PUBLIC_DEMO_SITE_KEY",
            "NEXT_PUBLIC_DEMO_PUBLIC_KEY",
        )
        print(
            json.dumps(
                {key: "" if args[key] is None else str(args[key]) for key in keys},
                sort_keys=True,
            )
        )
    elif mode == "redis-limit":
        redis = services["redis"]
        limits = [
            memory_bytes(redis.get("environment", {}).get("REDIS_MAXMEMORY", "512mb")),
            memory_bytes(redis.get("mem_limit", "1g")),
        ]
        if min(limits) <= 0:
            raise ValueError("Redis memory limits must be positive")
        print(min(limits))
    elif mode == "images":
        project = config["name"]
        for name in sys.argv[2:]:
            print(services[name].get("image", f"{project}-{name}"))
    else:
        raise ValueError("Unknown compose config operation")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        # Only our fixed diagnostics are printed; JSON/OS errors never include env contents.
        message = (
            str(exc)
            if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError)
            else "Unable to read deployment configuration"
        )
        print(message, file=sys.stderr)
        sys.exit(1)
