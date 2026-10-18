"""polars/spill_stats.py: a stored meter record -> the query's record. Needs no
GPU library: the record half of the module imports none.

The fixture is SF1 q9 of `make spill-smoke` (128 MiB device spill limit,
pinned spill): q9/meter.json, the warm and timed passes' meters.
"""
import json
import os
import shutil
import tempfile
import unittest

from tests import load
import spill_results as sr

HERE = os.path.dirname(os.path.abspath(__file__))
ps = load("polars/spill_stats.py", "polars_spill_stats")

FIX = os.path.join(HERE, "fixtures", "spill", "polars")


class Record(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ev = os.path.join(self.tmp.name, "polars_gpu")
        shutil.copytree(FIX, self.ev)
        open(os.path.join(self.ev, "records.jsonl"), "w").close()
        self.a = type("A", (), dict(q_dir=os.path.join(self.ev, "q9"), sf=1, query=9, evidence_dir=self.ev))

    def tearDown(self):
        self.tmp.cleanup()

    def meter(self):
        with open(os.path.join(self.ev, "q9", "meter.json")) as f:
            return json.load(f)

    def test_record(self):
        ps.cmd_record(self.a)
        rec = sr.read_records(self.ev)["query9"]
        self.assertEqual(sr.validate(rec), [])
        self.assertNotIn("error", rec)
        self.assertTrue(all(sr.check_result(c) for c in rec["checks"]))
        rows = {r["kind"]: r for r in rec["rows"]}
        t = self.meter()["timed"]
        self.assertTrue(t["pinned"])
        self.assertEqual(t["spill_d2pinned_bytes"], t["spill_d2h_bytes"])     # all of it pinned
        self.assertEqual(rows["spill_d2h"]["logical_b"], t["spill_d2h_bytes"])
        self.assertEqual(rows["restore_h2d"]["logical_b"], t["restore_h2d_bytes"])
        self.assertEqual((rows["peak_host"]["quality"], rows["peak_host"]["logical_b"]),
                         ("LIFETIME", t["pinned_peak_lifetime_bytes"]))
        h = sr.headline("polars_gpu", rec["rows"])
        self.assertIsNone(h["disk_spill"])                                    # no disk tier: "—"
        self.assertEqual(h["pressure_spill"][0], t["spill_d2h_bytes"])

    def test_meter_variants(self):
        """Pageable spill has no host peak; a collect off the spill-limited
        resource fails its check; the spill limit parses."""
        t = dict(self.meter()["timed"], pinned=False, pinned_peak_lifetime_bytes=None)
        rows, _ = ps.rows(t, "OK")
        self.assertEqual({r["kind"]: r for r in rows}["peak_host"]["quality"], "NA")
        rows, checks = ps.rows(dict(self.meter()["timed"], mr_ok=False), "OK")
        failed = [c["name"] for c in checks if not sr.check_result(c)]
        self.assertEqual(failed, ["collects on the spill-limited resource"])
        self.assertEqual(ps._bytes("128MiB"), 128 << 20)
        self.assertEqual(ps._bytes(str(22 << 30)), 22 << 30)
        self.assertIsNone(ps._bytes(None))

    def test_no_timed_meter_gives_the_na_stub(self):
        m = self.meter()
        with open(os.path.join(self.ev, "q9", "meter.json"), "w") as f:
            json.dump({"warm": m["warm"]}, f)
        ps.cmd_record(self.a)
        rec = sr.read_records(self.ev)["query9"]
        self.assertEqual(sr.validate(rec), [])
        self.assertIn("no timed pass", rec["error"])


if __name__ == "__main__":
    unittest.main()
