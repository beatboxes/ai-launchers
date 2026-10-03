"""Google Application Default Credentials for the ``vertex`` Gemini target (DESIGN §4).

ADC file: ``options["adc_path"]`` | ``$GOOGLE_APPLICATION_CREDENTIALS`` |
``$CLOUDSDK_CONFIG/application_default_credentials.json`` | ``%APPDATA%\\gcloud\\…`` (Windows) |
``~/.config/gcloud/application_default_credentials.json``.

* ``type == "authorized_user"`` (``gcloud auth application-default login``): form POST
  ``grant_type=refresh_token`` + ``client_id``/``client_secret``/``refresh_token`` to
  ``https://oauth2.googleapis.com/token`` (``AI_GATEWAY_GOOGLE_TOKEN_URL``); the access token is
  cached in memory until ``expires_in - 300`` s; the file is never written.
* any other type, or no file but ``gcloud`` on PATH: ``gcloud auth application-default
  print-access-token`` (list args, 60 s timeout, ``.cmd`` shim on Windows), cached 45 minutes.
* ``project()``: ``options["project"]`` > ``GOOGLE_CLOUD_PROJECT`` / ``CLOUDSDK_CORE_PROJECT`` /
  ``GCLOUD_PROJECT`` > ADC ``quota_project_id`` > ``gcloud config get-value project`` (cached) >
  ``AuthError`` ("set GOOGLE_CLOUD_PROJECT"). ``location()``: ``options["location"]`` >
  ``GOOGLE_CLOUD_LOCATION`` > ``"global"``.
* Headers: ``Authorization: Bearer`` + ``x-goog-user-project`` (when a project is known).
"""

import json
import os
import shutil
import subprocess
import time
from urllib.parse import urlencode

from ..compat import rfc3339_format
from ..transport import HttpClient, TransportError
from .base import AuthError, RefreshingAuth, Token
from .codex_chatgpt import TERMINAL_OAUTH_ERRORS, oauth_error_code

__all__ = ["GcloudADCAuth", "GOOGLE_TOKEN_URL", "ADC_FILE", "adc_path_for", "PROJECT_ENV_VARS"]

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
ADC_FILE = "application_default_credentials.json"
PROJECT_ENV_VARS = ("GOOGLE_CLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT", "GCLOUD_PROJECT")
GCLOUD_TOKEN_TTL = 45 * 60.0
GCLOUD_TIMEOUT = 60
HINT = "run `gcloud auth application-default login`"
_UNSET = object()


def adc_path_for(environ, windows=None, home=None):
    """The ADC file path gcloud / Google client libraries would use for ``environ``."""
    windows = (os.name == "nt") if windows is None else windows
    explicit = environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        return os.path.expanduser(explicit)
    config = environ.get("CLOUDSDK_CONFIG")
    if config:
        return os.path.join(os.path.expanduser(config), ADC_FILE)
    if windows and environ.get("APPDATA"):
        return os.path.join(environ["APPDATA"], "gcloud", ADC_FILE)
    home = home or environ.get("USERPROFILE" if windows else "HOME") or os.path.expanduser("~")
    return os.path.join(home, ".config", "gcloud", ADC_FILE)


class GcloudADCAuth(RefreshingAuth):
    """``gcloud_adc`` auth kind (see module docstring)."""

    kind = "gcloud_adc"

    def __init__(self, provider_id="", options=None, environ=None, http=None):
        RefreshingAuth.__init__(self, provider_id)
        self.options = dict(options or {})
        self._environ = environ
        self._http = http
        self._project = _UNSET

    # ---- environment -------------------------------------------------------------------
    def _env(self):
        return os.environ if self._environ is None else self._environ

    def adc_path(self):
        explicit = self.options.get("adc_path")
        return os.path.expanduser(explicit) if explicit else adc_path_for(self._env())

    def _read_adc(self):
        """Parsed ADC JSON (dict) or None (missing/unreadable/malformed)."""
        try:
            with open(self.adc_path(), "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def gcloud_path(self):
        return shutil.which("gcloud", path=self._env().get("PATH"))

    def _run_gcloud(self, args):
        """``(rc, stdout, stderr)`` of ``gcloud <args>``; None if gcloud is missing."""
        exe = self.gcloud_path()
        if exe is None:
            return None
        try:
            r = subprocess.run([exe] + list(args), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=GCLOUD_TIMEOUT,
                               env=None if self._environ is None else dict(self._environ),
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            raise AuthError("`gcloud %s` timed out after %ds" % (" ".join(args), GCLOUD_TIMEOUT), HINT, terminal=False)
        except OSError as exc:
            raise AuthError("cannot run gcloud: %s" % exc, HINT, terminal=False)
        return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")

    @staticmethod
    def _last_line(text):
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _is_user_creds(data):
        return bool(data and data.get("type") == "authorized_user" and
                    all(isinstance(data.get(k), str) and data.get(k)
                        for k in ("client_id", "client_secret", "refresh_token")))

    def _mode(self):
        """``"authorized_user"`` | ``"gcloud"`` | None (no usable credential source)."""
        if self._is_user_creds(self._read_adc()):
            return "authorized_user"
        return "gcloud" if self.gcloud_path() else None

    # ---- RefreshingAuth hooks ------------------------------------------------------------
    def _load(self):
        mode = self._mode()
        if mode is None:
            raise AuthError("no Google credentials (no authorized_user ADC file at %s and `gcloud` is not on PATH)"
                            % self.adc_path(), HINT)
        return Token("", 0.0, {"mode": mode})  # placeholder: expired -> first headers() fetches a token

    def _refresh(self, token):
        data = self._read_adc()
        if self._is_user_creds(data):
            return self._refresh_user(data)
        if self.gcloud_path():
            return self._gcloud_token()
        raise AuthError("no Google credentials available", HINT)

    def _refresh_user(self, data):
        url = self._env().get("AI_GATEWAY_GOOGLE_TOKEN_URL") or GOOGLE_TOKEN_URL
        body = urlencode([("grant_type", "refresh_token"), ("client_id", data["client_id"]),
                          ("client_secret", data["client_secret"]),
                          ("refresh_token", data["refresh_token"])]).encode("ascii")
        if self._http is None:
            self._http = HttpClient(timeout=30.0, connect_timeout=15.0, environ=self._environ)
        try:
            resp = self._http.request("POST", url, {"Content-Type": "application/x-www-form-urlencoded",
                                                    "Accept": "application/json"}, body, stream=False)
        except TransportError as exc:
            raise AuthError("Google token refresh failed: %s" % exc.message, HINT, terminal=False)
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        if 200 <= resp.status < 300 and isinstance(payload, dict) and isinstance(payload.get("access_token"), str) \
                and payload["access_token"]:
            expires_in = payload.get("expires_in")
            ttl = float(expires_in) if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool) \
                else 3600.0
            return Token(payload["access_token"], time.time() + ttl, {"mode": "authorized_user"})
        code = oauth_error_code(payload)
        terminal = resp.status in (400, 401) or code in TERMINAL_OAUTH_ERRORS
        raise AuthError("Google token refresh failed (HTTP %d%s)" % (resp.status, (", " + code) if code else ""),
                        HINT, terminal=terminal)

    def _gcloud_token(self):
        res = self._run_gcloud(["auth", "application-default", "print-access-token"])
        if res is None:
            raise AuthError("`gcloud` is not on PATH", HINT)
        rc, out, err = res
        tok = self._last_line(out)
        if rc != 0 or not tok or " " in tok:
            tail = self._last_line(err)[:200]
            raise AuthError("`gcloud auth application-default print-access-token` failed (exit %d)%s"
                            % (rc, (": " + tail) if tail else ""), HINT)
        # +margin so the token is served for the full 45 minutes before the 300 s early refresh
        return Token(tok, time.time() + GCLOUD_TOKEN_TTL + self.refresh_margin, {"mode": "gcloud"})

    def _token_headers(self, token):
        h = {"Authorization": "Bearer " + token.access_token}
        project = self._resolve_project()
        if project:
            h["x-goog-user-project"] = project
        return h

    def relogin_hint(self):
        return HINT

    # ---- project / location --------------------------------------------------------------
    def _resolve_project(self, allow_gcloud=True):
        with self._lock:
            if self._project is not _UNSET:
                return self._project
            env = self._env()
            project = self.options.get("project") or next((env[v] for v in PROJECT_ENV_VARS if env.get(v)), None)
            if not project:
                quota = (self._read_adc() or {}).get("quota_project_id")
                project = quota if isinstance(quota, str) and quota else None
            if not project and not allow_gcloud:
                return None
            if not project:
                res = None
                try:
                    res = self._run_gcloud(["config", "get-value", "project"])
                except AuthError:
                    res = None
                if res is not None and res[0] == 0:
                    value = self._last_line(res[1])
                    project = value if value and value != "(unset)" and " " not in value else None
            self._project = project
            return project

    def project(self):
        """Google Cloud project id; ``AuthError`` when none can be determined."""
        project = self._resolve_project()
        if not project:
            raise AuthError("no Google Cloud project configured",
                            "set GOOGLE_CLOUD_PROJECT (or run `gcloud config set project <id>`)")
        return project

    def location(self):
        return self.options.get("location") or self._env().get("GOOGLE_CLOUD_LOCATION") or "global"

    def describe(self):
        mode = None
        try:
            mode = self._mode()
        except AuthError:
            pass
        d = {"kind": self.kind, "provider": self.provider_id, "available": mode is not None,
             "source": {"authorized_user": "adc authorized_user", "gcloud": "gcloud print-access-token"}.get(mode),
             "adc_path": self.adc_path(), "project": self._resolve_project(allow_gcloud=False),
             "location": self.location()}
        tok = self._token
        if tok is not None and tok.access_token and tok.expires_at:
            d["expires_at"] = rfc3339_format(tok.expires_at, "seconds")
        if mode is None:
            d["hint"] = HINT
        return d
