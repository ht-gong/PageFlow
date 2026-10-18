"""rapids/eventlog_stats.py: the Spark event log (+ runner.json, monitor.csv,
the driver log) -> the record of the timed run.

The fixture is SF1 q9 of `make spill-smoke` (256 MiB RMM pool, 16 MiB host
spill store, 8 shuffle partitions), its event log cut down to the events and
fields the parser reads (pyarrow writes it back as zstd).
"""
import json
import os
import shutil
import tempfile
import unittest

from tests import load
import spill_results as sr

HERE = os.path.dirname(os.path.abspath(__file__))
es = load("rapids/eventlog_stats.py", "rapids_eventlog_stats")
FIX = os.path.join(HERE, "fixtures", "spill", "rapids")
TIMED = {"broadcast_logical_bytes": 18245888, "shuffle_d2h_logical_bytes": 37555240,
         "shuffle_h2d_logical_bytes": 37731027, "shuffle_write_wire_bytes": 19238921,
         "shuffle_read_wire_bytes": 19273148, "scan_input_wire_bytes": 122509361,
         "n_shuffle_exchanges": 4, "n_broadcast_exchanges": 4, "spill_gpu_to_host_logical_bytes": 0}


class Parse(unittest.TestCase):
    def test_event_log_and_driver_log(self):
        """The timed pass apart from the warm pass; the driver log's facts, its
        store-spilled lines cut to the pass."""
        rec, meta = es.parse(es.event_log(os.path.join(FIX, "q9")))
        timed = rec[("timed", 9)]
        for k, v in TIMED.items():
            self.assertEqual(timed[k], v, k)
        self.assertIn(("warm", 9), rec)
        self.assertGreater(rec[("warm", 9)]["n_tasks"], 0)
        self.assertTrue(meta["app_end"])
        self.assertEqual(meta["bad_lines"], 0)
        self.assertEqual(meta["spark_version"], "3.5.8")
        self.assertEqual(meta["spark_props"]["spark.sql.shuffle.partitions"], "8")
        with open(os.path.join(FIX, "q9", "runner.json")) as f:
            p = json.load(f)["passes"]["timed:q9"]
        d = es.driver_facts(os.path.join(FIX, "q9", "driver.log"), p["t0"], p["t1"])
        self.assertEqual((d["rmm_pool"], d["rmm_pool_b"]), ("ASYNC", 256 << 20))
        self.assertEqual((d["dropped_events"], d["store_spilled_b"]), (0, 0))
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write("26/10/09 07:20:01 INFO DeviceMemoryEventHandler: Device allocation of 5 bytes failed. "
                    "Device store spilled 100 bytes. First attempt.\n"
                    "26/10/09 07:20:09 INFO DeviceMemoryEventHandler: Device allocation of 5 bytes failed. "
                    "Device store spilled 7 bytes. First attempt.\n")
        try:
            from datetime import datetime
            t = datetime.strptime("26/10/09 07:20:09", "%y/%m/%d %H:%M:%S").timestamp()
            d = es.driver_facts(f.name, t - 0.5, t + 0.2)
            self.assertEqual((d["store_spilled_b"], d["store_spilled_lines"]), (7, 1))
        finally:
            os.remove(f.name)


class Record(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ev = os.path.join(self.tmp.name, "rapids")
        shutil.copytree(FIX, self.ev)
        open(os.path.join(self.ev, "records.jsonl"), "w").close()
        self.q = os.path.join(self.ev, "q9")

    def tearDown(self):
        self.tmp.cleanup()

    def record(self):
        a = type("A", (), dict(q_dir=self.q, sf=1, query=9, evidence_dir=self.ev))
        es.cmd_record(a)
        return sr.read_records(self.ev)["query9"]

    def test_record(self):
        rec = self.record()
        self.assertEqual(sr.validate(rec), [])
        self.assertNotIn("error", rec)
        self.assertTrue(all(sr.check_result(c) for c in rec["checks"]))
        rows = {r["kind"]: r for r in rec["rows"]}
        self.assertEqual(rows["broadcast"]["logical_b"], 18245888)
        self.assertEqual((rows["shuffle_write"]["logical_b"], rows["shuffle_write"]["wire_b"]), (None, 19238921))
        self.assertEqual(rows["spill_disk_files"]["quality"], "LOWER_BOUND")
        self.assertEqual(rows["peak_gpu"]["quality"], "OK")
        h = sr.headline("rapids", rec["rows"])
        self.assertEqual(h["off_gpu"][0], 37555240 + 18245888)
        self.assertEqual(h["design_disk"], (None, 19238921, "OK"))             # wire only
        self.assertEqual(h["read_back"][2], "LOWER_BOUND")                    # restore bytes are not counted

    def test_truncated_logs(self):
        """A killed JVM leaves an .inprogress log cut mid-line (PARTIAL rows),
        or a zstd frame cut short (read up to its end)."""
        import pyarrow as pa
        log = es.event_log(self.q)
        with open(log, "rb") as f:
            raw = f.read()
        with pa.CompressedInputStream(pa.OSFile(log), "zstd") as f:
            data = f.read()
        with self.subTest("cut mid-line"):
            os.remove(log)
            with open(os.path.join(self.q, "local-1.inprogress"), "wb") as f:
                f.write(data[: int(len(data) * 0.97)])       # inside the timed pass, mid-line
            rec, meta = es.parse(es.event_log(self.q))
            self.assertFalse(meta["app_end"])
            self.assertLessEqual(meta["bad_lines"], 1)
            self.assertIn(("timed", 9), rec)
            r = self.record()
            self.assertEqual(sr.validate(r), [])
            self.assertEqual({x["quality"] for x in r["rows"] if x["kind"] == "shuffle_d2h"}, {"PARTIAL"})
            os.remove(os.path.join(self.q, "local-1.inprogress"))
        with self.subTest("cut zstd frame"):
            with open(log, "wb") as f:
                f.write(raw[: len(raw) // 2])
            rec, meta = es.parse(log)
            self.assertFalse(meta["app_end"])
            self.assertTrue(rec)

    def test_no_event_log_gives_the_na_stub(self):
        os.remove(es.event_log(self.q))
        rec = self.record()
        self.assertEqual(sr.validate(rec), [])
        self.assertIn("no event log", rec["error"])


if __name__ == "__main__":
    unittest.main()
