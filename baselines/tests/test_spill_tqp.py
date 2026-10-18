"""../bench/tqp_spill_stats.py: a stored perf report -> the query's record.

The fixtures are SF1 q9 of the spill pass on the Vast box (bench/run_tqp.sh
with SPILL=1): tqp_vortex/ under --buffer_strategy cpu (q9/perf: the perf
report and cmdline.json), and tqp_vortex_smoke/ under SPILL_SMOKE=1's forced
pressure (the dataflow cache over a 4 MiB slot pool).

The builder lives beside its runner in bench/, which the baselines' container
does not mount: there these tests skip; run them on the host (stdlib only).
"""
import json
import os
import shutil
import tempfile
import unittest

from tests import BASELINES, load
import spill_results as sr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASELINES, "..")
BUILDER = os.path.join(REPO, "bench", "tqp_spill_stats.py")
FIX = os.path.join(HERE, "fixtures", "spill")
skip = unittest.skipUnless(os.path.exists(BUILDER), "bench/ is not mounted here: run on the host")
ts = load("../bench/tqp_spill_stats.py", "tqp_spill_stats") if os.path.exists(BUILDER) else None


class Evidence(unittest.TestCase):
    FIXTURE = "tqp_vortex"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ev = os.path.join(self.tmp.name, "tqp_vortex")
        shutil.copytree(os.path.join(FIX, self.FIXTURE), self.ev)
        open(os.path.join(self.ev, "records.jsonl"), "w").close()
        self.q = os.path.join(self.ev, "q9")
        self.a = type("A", (), dict(q_dir=self.q, sf=1, query=9, evidence_dir=self.ev))

    def tearDown(self):
        self.tmp.cleanup()

    def perf(self):
        with open(os.path.join(self.q, "perf", "SF1_9.json")) as f:
            return json.load(f)

    def write_perf(self, perf):
        with open(os.path.join(self.q, "perf", "SF1_9.json"), "w") as f:
            json.dump(perf, f)

    def record(self):
        ts.cmd_record(self.a)
        return sr.read_records(self.ev)["query9"]

    def failed_after(self, edit):
        """The checks that fail once edit(kinds) has changed the report's
        query-phase ledger (on a fresh copy of the fixture)."""
        self.tearDown(); self.setUp()
        perf = self.perf()
        edit(perf["movement"]["kinds"])
        self.write_perf(perf)
        return [c["name"] for c in self.record()["checks"] if not sr.check_result(c)]


@skip
class CpuRecord(Evidence):
    def test_record(self):
        """The ledger's kinds as they are, the sums and peaks, every check
        holding; comparable with the other engines' (everything off the GPU is
        by design, no pressure spill, no disk tier)."""
        mv_mod = load("../IO/telemetry/movement.py", "tqp_movement")
        self.assertEqual(set(ts.LEDGER) | {"result_d2h"}, set(mv_mod.KINDS))
        self.assertLessEqual(set(ts.LEDGER) | {"result_d2h", "born", "output_read"},
                             set(sr.KINDS["tqp_vortex"]))
        self.assertEqual(set(ts.READS), set(mv_mod.READ_KINDS))
        cat = sr.KINDS["tqp_vortex"]
        self.assertEqual({k for k, c in cat.items() if c in sr.HEADLINES["off_gpu"]},
                         set(mv_mod.SPILL_KINDS))
        # Cleaning runs only while pressure holds its gate open.
        self.assertEqual({k for k, c in cat.items() if c in sr.HEADLINES["pressure_spill"]},
                         {"spill_evict", "spill_rejected", "spill_mirror"})

        rec = self.record()
        self.assertEqual(sr.validate(rec), [])
        self.assertNotIn("error", rec)
        names = [c["name"] for c in rec["checks"]]
        self.assertTrue(all(sr.check_result(c) for c in rec["checks"]), rec["checks"])
        self.assertIn("cpu: born = spill_bypass", names)
        self.assertIn("arena peak >= every pipe's peak", names)
        rows = {r["kind"]: r for r in rec["rows"]}
        mv, out = self.perf()["movement"]["kinds"], self.perf()["movement"]["output"]
        for k in ts.LEDGER:
            self.assertEqual((rows[k]["logical_b"], rows[k]["wire_b"], rows[k]["count"]),
                             (mv[k]["logical_b"], mv[k]["wire_b"],
                              None if k in ts.UVA else mv[k]["pages"]), k)
        # Zero-copy reads have bytes and no pages; the sums have no count.
        self.assertGreater(rows["scan_uva"]["logical_b"], 0)
        self.assertIsNone(rows["scan_uva"]["count"])
        self.assertIsNone(rows["born"]["count"])
        self.assertIsNone(rows["output_read"]["count"])
        self.assertEqual(rows["born"]["logical_b"], mv["spill_bypass"]["logical_b"])
        self.assertEqual(rows["result_d2h"]["wire_b"], out["result_d2h"]["wire_b"])
        self.assertEqual(rows["result_d2h"]["count"], 3)                      # q9's three columns
        res = self.perf()["residency"]
        self.assertEqual(rows["peak_gpu_arena"]["logical_b"], res["gpu_arena"]["peak_b"])
        self.assertEqual(rows["peak_host_intermediate"]["logical_b"],
                         res["host_pool"]["peak_b"]["intermediate"])
        self.assertEqual(rows["slot_slab"]["logical_b"], 0)                   # no cache under cpu
        h = sr.headline("tqp_vortex", rec["rows"])
        self.assertEqual(h["off_gpu"][:2], (mv["spill_bypass"]["logical_b"], mv["spill_bypass"]["wire_b"]))
        self.assertEqual(h["pressure_spill"], (0, 0, "OK"))
        self.assertIsNone(h["disk_spill"])
        self.assertIsNone(h["design_disk"])
        self.assertEqual(h["read_back"][0], mv["restore_h2d"]["logical_b"] + mv["restore_uva"]["logical_b"])
        # Cache traffic under cpu breaks its identity.
        self.assertEqual(self.failed_after(lambda k: k.update(
            spill_evict={"pages": 1, "logical_b": 9, "wire_b": 9})), ["cpu: spill_evict = 0"])

    def test_a_failed_or_missing_report(self):
        """A failed query keeps its partial report; no report, or one without
        the telemetry, gives the all-NA stub with the pass's status."""
        with self.subTest("partial"):
            perf = self.perf()
            perf["wall"] = {"occurrence": 0, "approx_time": None, "status": "FAIL",
                            "time_to_failure": 0.05, "error": "MemoryError: pinned host pool exhausted"}
            self.write_perf(perf)
            with open(os.path.join(self.ev, "pass.csv"), "a") as f:
                f.write("query9,FAIL,0.050,MemoryError: pinned host pool exhausted\n")
            rec = self.record()
            self.assertEqual(sr.validate(rec), [])
            self.assertEqual(rec["run_status"], "FAIL")
            self.assertIn("pinned host pool", rec["error"])
            rows = {r["kind"]: r for r in rec["rows"]}
            self.assertEqual(rows["spill_bypass"]["quality"], "PARTIAL")
            self.assertNotIn("the result copy is counted", [c["name"] for c in rec["checks"]])
        with self.subTest("no report"):
            self.tearDown(); self.setUp()
            os.remove(os.path.join(self.q, "perf", "SF1_9.json"))
            with open(os.path.join(self.ev, "pass.csv"), "a") as f:
                f.write("query9,TIMEOUT,603,exceeded_query_timeout_603s\n")
            rec = self.record()
            self.assertEqual(sr.validate(rec), [])
            self.assertEqual(rec["run_status"], "TIMEOUT")
            self.assertIn("FileNotFoundError", rec["error"])
            self.assertTrue(all(r["quality"] == "NA" for r in rec["rows"] if r["kind"] != "pass_time"))
        with self.subTest("no telemetry"):
            self.tearDown(); self.setUp()
            perf = self.perf()
            del perf["residency"]
            self.write_perf(perf)
            self.assertIn("TQP_EXPERIMENT_TELEMETRY", self.record()["error"])

    def test_settings(self):
        """The pools as sized, the versions, and the queries whose CPU pin
        failed."""
        cmd = os.path.join(self.q, "perf", "cmdline.json")
        q18 = os.path.join(self.ev, "q18", "perf")
        os.makedirs(q18)
        for path, pin in ((cmd, {"node": 1, "n_cpus": 96, "error": None}),
                          (os.path.join(q18, "cmdline.json"),
                           {"node": None, "n_cpus": 0, "error": "[Errno 22]"})):
            with open(cmd) as f:
                c = json.load(f)
            with open(path, "w") as f:
                json.dump({**c, "cpu_pin": pin}, f)
        ts.cmd_settings(type("A", (), dict(evidence_dir=self.ev, sf=1, git="abc123"))())
        with open(os.path.join(self.ev, "settings.json")) as f:
            s = json.load(f)
        res = self.perf()["residency"]
        self.assertEqual(s["budgets"], {"gpu_b": res["gpu_arena"]["capacity_b"],
                                        "host_b": res["host_pool"]["capacity_b"], "disk_b": None})
        self.assertEqual(s["versions"]["tqp_vortex"], "abc123")
        self.assertEqual(s["effective"]["buffer_strategy"], "cpu")
        self.assertEqual(s["spill_dirs"], [])
        self.assertEqual((s["cpu_pin_nodes"], s["cpu_pin_failed"]), ([1], ["q18"]))


@skip
class CacheRecord(Evidence):
    FIXTURE = "tqp_vortex_smoke"

    def test_forced_pressure_spills_and_restores(self):
        rec = self.record()
        self.assertEqual(sr.validate(rec), [])
        self.assertTrue(all(sr.check_result(c) for c in rec["checks"]), rec["checks"])
        names = [c["name"] for c in rec["checks"]]
        self.assertIn("cache: clean reclaims were mirrored first", names)
        self.assertIn("cache: decoded bytes home <= decoded bytes filled", names)
        self.assertIn("cache: bytes out of slots <= bytes filled", names)
        self.assertIn("cache: evicted reads only with copies home", names)
        self.assertNotIn("cpu: born = spill_bypass", names)
        rows = {r["kind"]: r for r in rec["rows"]}
        mv = self.perf()["movement"]["kinds"]
        h = sr.headline("tqp_vortex", rec["rows"])
        spills = ("spill_evict", "spill_rejected", "spill_mirror")
        pressure = [sum(mv[k][f] for k in spills) for f in ("logical_b", "wire_b")]
        self.assertGreater(pressure[0], 0)
        self.assertEqual(h["pressure_spill"], (*pressure, "OK"))
        # Every pressure kind counts decoded bytes, the granules' too: the
        # column's codecs stored them in fewer bytes.
        for k in spills:
            self.assertGreater(rows[k]["wire_b"], 0, k)
            self.assertGreater(rows[k]["logical_b"], rows[k]["wire_b"], k)
        self.assertGreater(rows["restore_evicted_h2d"]["wire_b"], 0)
        self.assertEqual(h["read_back"][0], sum(mv[k]["logical_b"] for k in
                                                ("restore_h2d", "restore_uva", "restore_evicted_h2d")))
        self.assertEqual(rows["slot_slab"]["logical_b"], 4 << 20)
        self.assertEqual(rows["born"]["logical_b"], sum(mv[k]["logical_b"] for k in
                                                        ("cache_fill", "spill_bypass", "spill_rejected")))

    def test_each_cache_identity(self):
        """Each identity the cache modes hold fails alone when broken; reading
        home bytes once per consumer (more than was spilled) breaks none."""
        zero = {"pages": 0, "logical_b": 0, "wire_b": 0}

        def more_home_than_filled(k):
            k["spill_evict"] = {"pages": k["cache_fill"]["pages"] + 1, "logical_b": 9, "wire_b": 9}

        def clean_never_mirrored(k):
            k["spill_mirror"] = dict(zero)
            k["avoided_clean"] = {"pages": 1, "logical_b": 9, "wire_b": 9}

        def more_out_of_slots(k):
            k["avoided_dead"]["wire_b"] += k["cache_fill"]["wire_b"]

        def evicted_reads_nothing_home(k):
            for n in ("spill_evict", "spill_rejected", "spill_mirror", "avoided_clean"):
                k[n] = dict(zero)

        def read_home_three_times(k):
            k["restore_evicted_h2d"]["wire_b"] = 3 * sum(
                k[n]["wire_b"] for n in ("spill_evict", "spill_rejected", "spill_mirror"))

        for edit, want in ((more_home_than_filled, ["cache: granules home <= granules filled"]),
                           (clean_never_mirrored, ["cache: clean reclaims were mirrored first"]),
                           (more_out_of_slots, ["cache: bytes out of slots <= bytes filled"]),
                           (evicted_reads_nothing_home, ["cache: evicted reads only with copies home"]),
                           (read_home_three_times, [])):
            with self.subTest(edit.__name__):
                self.assertEqual(self.failed_after(edit), want)


if __name__ == "__main__":
    unittest.main()
