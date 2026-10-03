"""auth.gcloud_adc: ADC path resolution, authorized_user refresh (cached, no write-back), terminal
errors, fake-gcloud print-access-token path (45 min cache), project resolution order (cached
`gcloud config get-value project`), location, headers, describe without secrets."""

import json
import os
import tempfile
import time
import unittest
from urllib.parse import parse_qs

from ._pkg import mod

gcloud = mod("auth.gcloud_adc")
auth_base = mod("auth.base")
auth_pkg = mod("auth")
config = mod("config")
mocks = mod("testing.mock_upstreams")
mock_auth = mod("testing.mock_auth")
fake_bins = mod("testing.fake_bins")

USER_ADC = {"type": "authorized_user", "client_id": mock_auth.GOOGLE_CLIENT_ID,
            "client_secret": mock_auth.GOOGLE_CLIENT_SECRET, "refresh_token": "g-rt",
            "quota_project_id": "quota-proj", "account": ""}


class GcloudADCTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = os.path.join(self.tmp.name, "gcloud")
        os.makedirs(self.cfg)
        self.adc = os.path.join(self.cfg, "application_default_credentials.json")
        self.bins = fake_bins.make_fake_bins(os.path.join(self.tmp.name, "bin"))
        self.empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(self.empty)
        self.srv = mocks.MockServer("google_oauth").start()
        self.env = {"PATH": self.bins, "CLOUDSDK_CONFIG": self.cfg,
                    "AI_GATEWAY_GOOGLE_TOKEN_URL": self.srv.url + "/token"}

    def tearDown(self):
        self.srv.stop()
        self.tmp.cleanup()

    def write_adc(self, data):
        with open(self.adc, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def auth(self, env=None, **options):
        return gcloud.GcloudADCAuth("vertex", options, environ=self.env if env is None else env)

    def gcloud_calls(self, *argv):
        return [c for c in fake_bins.read_calls(self.bins, "gcloud") if c["argv"][:len(argv)] == list(argv)]

    def test_adc_path_resolution(self):
        f = gcloud.adc_path_for
        name = gcloud.ADC_FILE
        self.assertEqual(f({"GOOGLE_APPLICATION_CREDENTIALS": "/k.json", "CLOUDSDK_CONFIG": "/c"}), "/k.json")
        self.assertEqual(f({"CLOUDSDK_CONFIG": "/c", "APPDATA": "/a"}, windows=True), os.path.join("/c", name))
        self.assertEqual(f({"APPDATA": "C:\\AppData", "USERPROFILE": "C:\\u"}, windows=True),
                         os.path.join("C:\\AppData", "gcloud", name))
        self.assertEqual(f({"HOME": "/home/u"}, windows=False), os.path.join("/home/u", ".config", "gcloud", name))
        self.assertEqual(f({"APPDATA": "/a", "HOME": "/home/u"}, windows=False),
                         os.path.join("/home/u", ".config", "gcloud", name))
        self.assertEqual(self.auth(adc_path="/explicit.json").adc_path(), "/explicit.json")
        self.assertEqual(self.auth().adc_path(), self.adc)

    def test_authorized_user_refresh_cached_no_write_back(self):
        self.write_adc(USER_ADC)
        mtime = os.stat(self.adc).st_mtime_ns
        a = self.auth()
        self.assertTrue(a.available())
        self.assertEqual(self.srv.requests, [], "available() never hits the network")
        h = a.headers()
        self.assertEqual(h, {"Authorization": "Bearer ya29.mock-1", "x-goog-user-project": "quota-proj"})
        form = parse_qs(self.srv.requests[-1]["body_raw"])
        self.assertEqual(form, {"grant_type": ["refresh_token"], "client_id": [mock_auth.GOOGLE_CLIENT_ID],
                                "client_secret": [mock_auth.GOOGLE_CLIENT_SECRET], "refresh_token": ["g-rt"]})
        self.assertEqual(a.headers(), h)
        self.assertEqual(mock_auth.refresh_count(self.srv), 1, "cached until expires_in - 300")
        self.assertAlmostEqual(a.current_token().expires_at, time.time() + 3599, delta=30)
        self.assertEqual(os.stat(self.adc).st_mtime_ns, mtime)
        with open(self.adc) as f:
            self.assertEqual(json.load(f), USER_ADC)
        self.assertEqual(self.gcloud_calls(), [], "gcloud never used for authorized_user ADC")
        d = a.describe()
        blob = json.dumps(d)
        for secret in ("ya29.mock-1", "g-rt", mock_auth.GOOGLE_CLIENT_SECRET):
            self.assertNotIn(secret, blob)
        self.assertEqual((d["source"], d["project"], d["location"]), ("adc authorized_user", "quota-proj", "global"))
        # on_unauthorized -> refresh once; stale `used` short-circuits
        self.assertTrue(a.on_unauthorized(h))
        self.assertEqual(a.headers()["Authorization"], "Bearer ya29.mock-2")
        self.assertTrue(a.on_unauthorized(h))
        self.assertEqual(mock_auth.refresh_count(self.srv), 2)

    def test_short_lived_tokens_refresh_each_time(self):
        self.srv.stop()
        self.srv = mocks.MockServer("google_oauth", options={"expires_in": 200}).start()
        self.env["AI_GATEWAY_GOOGLE_TOKEN_URL"] = self.srv.url + "/token"
        self.write_adc(USER_ADC)
        a = self.auth()
        a.headers()
        a.headers()
        self.assertEqual(mock_auth.refresh_count(self.srv), 2)

    def test_authorized_user_errors(self):
        self.write_adc(dict(USER_ADC, refresh_token="revoked"))
        with self.assertRaises(auth_base.AuthError) as cm:
            self.auth().headers()
        self.assertTrue(cm.exception.terminal)
        self.assertEqual(cm.exception.hint, "run `gcloud auth application-default login`")
        self.assertIn("invalid_grant", str(cm.exception))
        self.write_adc(dict(USER_ADC, client_secret="wrong"))
        with self.assertRaises(auth_base.AuthError) as cm:
            self.auth().headers()
        self.assertTrue(cm.exception.terminal)
        self.srv.stop()
        self.srv = mocks.MockServer("google_oauth", options={"fail_status": 503}).start()
        self.env["AI_GATEWAY_GOOGLE_TOKEN_URL"] = self.srv.url + "/token"
        self.write_adc(USER_ADC)
        a = self.auth()
        with self.assertRaises(auth_base.AuthError) as cm:
            a.headers()
        self.assertFalse(cm.exception.terminal)
        self.assertEqual(a.headers()["Authorization"], "Bearer ya29.mock-1")

    def test_other_adc_types_use_gcloud(self):
        self.write_adc({"type": "service_account", "project_id": "sa-proj", "private_key": "-----BEGIN"})
        a = self.auth()
        self.assertTrue(a.available())
        h = a.headers()
        self.assertEqual(h["Authorization"], "Bearer ya29.fake-gcloud-token")
        self.assertEqual(a.headers(), h)
        self.assertEqual(len(self.gcloud_calls("auth", "application-default", "print-access-token")), 1)
        self.assertAlmostEqual(a.current_token().expires_at - a.refresh_margin, time.time() + 45 * 60, delta=30)
        self.assertEqual(self.srv.requests, [])
        self.assertEqual(a.describe()["source"], "gcloud print-access-token")
        a.headers(force_refresh=True)
        self.assertEqual(len(self.gcloud_calls("auth", "application-default", "print-access-token")), 2)

    def test_no_file_uses_gcloud_then_nothing(self):
        a = self.auth()
        self.assertTrue(a.available())
        self.assertEqual(a.headers()["Authorization"], "Bearer ya29.fake-gcloud-token")
        self.assertEqual(a.headers()["x-goog-user-project"], "fake-gcloud-project")
        none = self.auth(env={"PATH": self.empty, "CLOUDSDK_CONFIG": self.cfg})
        self.assertFalse(none.available())
        with self.assertRaises(auth_base.AuthError) as cm:
            none.headers()
        self.assertEqual(cm.exception.hint, "run `gcloud auth application-default login`")
        self.assertEqual(none.describe()["hint"], "run `gcloud auth application-default login`")
        self.assertFalse(none.describe()["available"])

    def test_gcloud_failure(self):
        fake_bins.set_behaviour(self.bins, gcloud_token_rc=1)
        with self.assertRaises(auth_base.AuthError) as cm:
            self.auth().headers()
        self.assertIn("application-default login", str(cm.exception))
        fake_bins.set_behaviour(self.bins, gcloud_token_rc=0, gcloud_token="")
        with self.assertRaises(auth_base.AuthError):
            self.auth().headers()

    def test_project_resolution_order(self):
        self.write_adc(USER_ADC)
        base = dict(self.env)
        envs = {"GOOGLE_CLOUD_PROJECT": "p-google", "CLOUDSDK_CORE_PROJECT": "p-core", "GCLOUD_PROJECT": "p-gcloud"}
        self.assertEqual(self.auth(dict(base, **envs), project="p-option").project(), "p-option")
        self.assertEqual(self.auth(dict(base, **envs)).project(), "p-google")
        del envs["GOOGLE_CLOUD_PROJECT"]
        self.assertEqual(self.auth(dict(base, **envs)).project(), "p-core")
        del envs["CLOUDSDK_CORE_PROJECT"]
        self.assertEqual(self.auth(dict(base, **envs)).project(), "p-gcloud")
        self.assertEqual(self.auth(base).project(), "quota-proj")
        self.assertEqual(self.gcloud_calls(), [])
        self.write_adc(dict(USER_ADC, quota_project_id=None))
        a = self.auth(base)
        self.assertEqual(a.project(), "fake-gcloud-project")
        self.assertEqual(a.project(), "fake-gcloud-project")
        self.assertEqual(a.headers()["x-goog-user-project"], "fake-gcloud-project")
        self.assertEqual(len(self.gcloud_calls("config", "get-value", "project")), 1, "cached")
        fake_bins.set_behaviour(self.bins, gcloud_project=None)
        unset = self.auth(base)
        with self.assertRaises(auth_base.AuthError) as cm:
            unset.project()
        self.assertIn("GOOGLE_CLOUD_PROJECT", str(cm.exception))
        self.assertNotIn("x-goog-user-project", unset.headers())
        nogcloud = self.auth(dict(base, PATH=self.empty))
        with self.assertRaises(auth_base.AuthError):
            nogcloud.project()
        self.assertIsNone(nogcloud.describe()["project"])

    def test_location(self):
        self.assertEqual(self.auth().location(), "global")
        self.assertEqual(self.auth(dict(self.env, GOOGLE_CLOUD_LOCATION="us-east5")).location(), "us-east5")
        self.assertEqual(self.auth(dict(self.env, GOOGLE_CLOUD_LOCATION="us-east5"), location="europe-west4")
                         .location(), "europe-west4")

    def test_make_auth(self):
        a = auth_pkg.make_auth({"kind": "gcloud_adc", "project": "p", "location": "us-central1"},
                               config.SecretStore(), "gemini-vertex")
        self.assertIsInstance(a, gcloud.GcloudADCAuth)
        self.assertEqual((a.project(), a.location(), a.kind), ("p", "us-central1", "gcloud_adc"))


if __name__ == "__main__":
    unittest.main()
