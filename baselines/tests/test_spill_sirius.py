"""sirius/spill_stats.py: the per-query config, quent + DEBUG log -> the record.

The fixture is SF1 q9 of `make spill-smoke` (768 MiB GPU budget, 64 MiB host
tier): the quent kinds the builder reads, the gzipped DEBUG log and the CLI's
stdout with the /proc brackets.
"""
import json
import os
import shutil
import tempfile
import time
import unittest

from tests import load
import spill_results as sr

HERE = os.path.dirname(os.path.abspath(__file__))
ss = load("sirius/spill_stats.py", "sirius_spill_stats")

FIX = os.path.join(HERE, "fixtures", "spill", "sirius")
CONFIG = os.path.join(HERE, "..", "sirius", "sirius.yaml")
_TZ = os.environ.get("TZ")


def _set_tz(tz):
    if tz is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = tz
    time.tzset()


def setUpModule():
    # The fixture's log was written on a box in UTC; spdlog writes local time.
    _set_tz("UTC")


def tearDownModule():
    _set_tz(_TZ)


class LogTime(unittest.TestCase):
    def tearDown(self):
        _set_tz("UTC")

    def test_log_time_is_the_local_time_of_the_process(self):
        ts = "2026-10-09 12:00:00.250"
        utc = ss.to_ns(ts)
        self.assertEqual(utc, 1791547200.25e9)
        _set_tz("America/New_York")    # EDT, UTC-4, in October
        self.assertEqual(ss.to_ns(ts) - utc, 4 * 3600e9)


class Config(unittest.TestCase):
    def test_quent_block_smoke_budgets_and_sizes(self):
        with open(CONFIG) as f:
            base = f.read()
        out = ss.yaml_set(base, {"sirius.telemetry.enable_quent": "true",
                                 "sirius.telemetry.output_directory": '"/q/quent"', **ss.SMOKE})
        get = lambda p: ss.yaml_get(out, p)
        self.assertEqual(get("sirius.telemetry.enable_quent"), "true")
        self.assertEqual(get("sirius.telemetry.output_directory"), "/q/quent")
        self.assertIsNone(get("sirius.memory.gpu.usage_limit_fraction"))
        self.assertEqual(get("sirius.memory.gpu.usage_limit_bytes"), "768Mi")
        self.assertEqual(get("sirius.memory.host.capacity_bytes"), "64Mi")
        # untouched keys and every comment stay as they were
        self.assertEqual(get("sirius.memory.gpu.reservation_limit_fraction"),
                         ss.yaml_get(base, "sirius.memory.gpu.reservation_limit_fraction"))
        self.assertEqual(get("sirius.operator_params.max_build_hash_table_bytes"),
                         ss.yaml_get(base, "sirius.operator_params.max_build_hash_table_bytes"))
        self.assertEqual([ln for ln in base.splitlines() if ln.lstrip().startswith("#")],
                         [ln for ln in out.splitlines() if ln.lstrip().startswith("#")])
        # new keys land inside their block, at its indentation
        lines = out.splitlines()
        i = lines.index("  telemetry:")
        self.assertTrue(all(ln.startswith("    ") for ln in lines[i + 1:] if ln.strip()))
        # a key whose block is missing is an error
        with self.assertRaises(ValueError):
            ss.yaml_set("sirius:\n  memory:\n    gpu: {}\n", {"sirius.telemetry.exporter": "ndjson"})
        self.assertEqual(ss.size_bytes("128Gi"), 128 << 30)
        self.assertEqual(ss.size_bytes("40GB"), 40 * 10**9)
        self.assertEqual(ss.size_bytes("805306368"), 805306368)
        self.assertIsNone(ss.size_bytes("lots"))



class Record(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ev = os.path.join(self.tmp.name, "sirius")
        shutil.copytree(FIX, self.ev)

    def tearDown(self):
        self.tmp.cleanup()

    def cmd_record(self):
        a = type("A", (), dict(q_dir=os.path.join(self.ev, "q9"), sf=1, query=9, evidence_dir=self.ev))
        open(os.path.join(self.ev, "records.jsonl"), "w").close()
        ss.cmd_record(a)
        return sr.read_records(self.ev)["query9"]

    def test_record(self):
        """quent's numbers, the DEBUG log agreeing to the byte, and the record
        keeping the contract and its identities."""
        n = ss.native(os.path.join(self.ev, "q9"), 9)
        self.assertTrue(n["complete"])
        self.assertEqual(n["label"], "timed_q9")
        # plan §3.3: the same four numbers as the exploration's SF1 probe
        self.assertEqual(n["spill_gpu_host_b"], 46915188)
        self.assertEqual(n["spill_host_disk_b"], 18461696)
        self.assertEqual(n["restore_host_gpu_b"], 28453492)
        self.assertEqual(n["restore_disk_gpu_b"], 18467520)
        self.assertEqual(n.get("spill_gpu_disk_b", 0), 0)
        # the DEBUG log agrees to the byte; the host monitor drove host -> disk
        self.assertEqual(n["log_gpu_to_host_b"], n["spill_gpu_host_b"])
        self.assertEqual(n["log_host_to_disk_b"], n["spill_host_disk_b"])
        self.assertEqual(n["log_host_monitor_b"], 18461696)
        # the parquet read: rchar of the timed run minus the disk-tier restores
        self.assertEqual(n["scan_read_b"], n["io_rchar_b"] - n["restore_disk_gpu_b"])
        rec = self.cmd_record()
        self.assertEqual(sr.validate(rec), [])
        self.assertNotIn("error", rec)
        self.assertEqual(rec["run_status"], "OK")
        self.assertTrue(rec["checks"] and all(sr.check_result(c) for c in rec["checks"]))
        rows = {r["kind"]: r for r in rec["rows"]}
        self.assertEqual(rows["pool_peak_gpu"]["quality"], "LIFETIME")
        h = sr.headline("sirius", rec["rows"])
        self.assertEqual(h["pressure_spill"][0], 46915188)
        self.assertEqual(h["disk_spill"][0], 18461696)
        self.assertEqual(h["read_back"][0], 28453492 + 18467520)
        # the raw evidence is compressed, and is still readable
        self.assertTrue(os.path.exists(os.path.join(self.ev, "q9", "record.json")))
        self.assertEqual(ss.native(os.path.join(self.ev, "q9"), 9)["spill_gpu_host_b"], 46915188)

    def test_killed_or_without_quent(self):
        """A killed process's rows are PARTIAL; no quent session gives the
        all-NA stub."""
        n = ss.native(os.path.join(self.ev, "q9"), 9)
        rows, _ = ss.rows(n, "KILLED_DISK")
        self.assertEqual({r["quality"] for r in rows if r["kind"].startswith("spill_")}, {"PARTIAL"})
        shutil.rmtree(os.path.join(self.ev, "q9", "quent"))
        rec = self.cmd_record()
        self.assertEqual(sr.validate(rec), [])
        self.assertIn("no quent session", rec["error"])
        self.assertEqual({r["quality"] for r in rec["rows"] if r["kind"] != "pass_time"}, {"NA"})


if __name__ == "__main__":
    unittest.main()
