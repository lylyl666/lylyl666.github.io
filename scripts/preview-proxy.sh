#!/usr/bin/env bash
set -euo pipefail

port="${1:-4000}"

if [[ -n "${VSCODE_PROXY_URI:-}" ]]; then
  baseurl="$(
    PREVIEW_PORT="$port" python - <<'PY'
import os
from urllib.parse import urlparse

uri = os.environ["VSCODE_PROXY_URI"].replace("{{port}}", os.environ["PREVIEW_PORT"])
print(urlparse(uri).path.rstrip("/"))
PY
  )"

  bundle exec jekyll build --baseurl "$baseurl"
  exec python -m http.server "$port" --bind 0.0.0.0 --directory _site
fi

exec bundle exec jekyll serve --host 0.0.0.0 --port "$port"
