#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="${SCRIPT_DIR}/robot_server.service.template"
ROBOT_SERVER_PATH="${SCRIPT_DIR}/robot_server.py"

SERVICE_NAME="${SERVICE_NAME:-robot_server}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(id -un)}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
WORKING_DIRECTORY="${WORKING_DIRECTORY:-${SCRIPT_DIR}}"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ ! -f "${TEMPLATE_PATH}" ]]; then
  echo "[robot-server] missing template: ${TEMPLATE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ROBOT_SERVER_PATH}" ]]; then
  echo "[robot-server] missing script: ${ROBOT_SERVER_PATH}" >&2
  exit 1
fi

TMP_UNIT="$(mktemp)"

"${PYTHON_BIN}" - "${TEMPLATE_PATH}" "${TMP_UNIT}" "${RUN_USER}" "${WORKING_DIRECTORY}" "${PYTHON_BIN}" "${ROBOT_SERVER_PATH}" "${HOST}" "${PORT}" <<'PY'
from pathlib import Path
import sys

template_path, output_path, run_user, working_directory, python_bin, script_path, host, port = sys.argv[1:]
text = Path(template_path).read_text(encoding="utf-8")
replacements = {
    "__RUN_USER__": run_user,
    "__WORKING_DIRECTORY__": working_directory,
    "__PYTHON_BIN__": python_bin,
    "__SCRIPT_PATH__": script_path,
    "__HOST__": host,
    "__PORT__": port,
}
for old, new in replacements.items():
    text = text.replace(old, new)
Path(output_path).write_text(text, encoding="utf-8")
PY

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
fi

echo "[robot-server] installing unit to ${UNIT_PATH}"
${SUDO} install -m 0644 "${TMP_UNIT}" "${UNIT_PATH}"
rm -f "${TMP_UNIT}"

echo "[robot-server] reloading systemd"
${SUDO} systemctl daemon-reload

echo "[robot-server] enabling ${SERVICE_NAME}"
${SUDO} systemctl enable "${SERVICE_NAME}"

echo "[robot-server] restarting ${SERVICE_NAME}"
${SUDO} systemctl restart "${SERVICE_NAME}"

echo "[robot-server] current status"
${SUDO} systemctl --no-pager --full status "${SERVICE_NAME}"
