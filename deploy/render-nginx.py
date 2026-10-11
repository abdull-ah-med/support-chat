#!/usr/bin/env python3
"""Render only validated host/certificate paths. Exit 3 when TLS must be left alone."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--http-only", action="store_true")
parser.add_argument("--hosts", nargs=3, metavar=("STAFF", "WIDGET", "MARKETING"))
args = parser.parse_args()
if args.hosts:
    hosts = args.hosts
else:
    hosts = subprocess.check_output(
        ["python3", "deploy/compose-config.py", "origins"], text=True
    ).splitlines()
if (
    len(hosts) != 3
    or len(set(hosts)) != 3
    or any(not re.fullmatch(r"[A-Za-z0-9.-]+", h) for h in hosts)
):
    sys.exit("Invalid nginx hosts")
text = Path("deploy/nginx.conf.template").read_text()
active = Path(os.getenv("NGINX_CONF_DEST", "/etc/nginx/conf.d/support-chat.conf"))
cert_root = Path(os.getenv("LETSENCRYPT_LIVE_ROOT", "/etc/letsencrypt/live"))
previous = active.read_text() if active.is_file() else ""
for label, host in zip(("STAFF", "WIDGET", "MARKETING"), hosts):
    text = text.replace(f"__{label}_HOST__", host)
    if args.http_only:
        continue
    # Bootstrap certbot -d ordering creates a combined marketing-host lineage.
    # Honor existing certbot lineage names (including -0001) when present.
    cert_dir = cert_root / hosts[2]
    for block in re.split(r"(?m)^\s*server\s*\{", previous):
        match = re.search(r"\bserver_name\s+([^;]+);", block)
        cert = re.search(r"\bssl_certificate\s+([^;]+);", block)
        if match and host in match[1].split() and cert:
            path = Path(cert[1].strip())
            if path.name == "fullchain.pem" and path.parent.parent == cert_root:
                cert_dir = path.parent
                break
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cert_dir.name):
        sys.exit("Invalid certificate lineage")
    if not all(
        (cert_dir / name).is_file() for name in ("fullchain.pem", "privkey.pem")
    ):
        print(
            "WARN: TLS files missing for "
            + label
            + "; keeping active nginx configuration",
            file=sys.stderr,
        )
        sys.exit(3)
    text = text.replace(f"__{label}_CERT__", str(cert_dir))
if args.http_only:
    text = re.sub(
        r"# BEGIN HTTPS REDIRECTS.*?# END HTTPS REDIRECTS", "", text, flags=re.DOTALL
    )
    text = re.sub(r"(?m)^\s*ssl_(?:certificate(?:_key)?|protocols)\s+.*\n", "", text)
    text = text.replace("listen 443 ssl;", "listen 80;")
print(text)
