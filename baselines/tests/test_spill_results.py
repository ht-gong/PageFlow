"""spill_results.py: the record contract, the merge, the headline sums.

  cd baselines && "$PY" -m unittest discover -s tests -t . -p "test_spill*.py"   (make spill-test)
"""
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import tests  # noqa: F401  (puts baselines/ on sys.path)
import spill_results as sr


def full_rows(engine, value=0):
    """Every kind of the engine, measured, plus pass_time."""
    rows = [sr.row(k, value, value, quality="OK", source="test") for k in sr.KINDS[engine]]
    return rows + [sr.row("pass_time", count=4, seconds=1.5, source="pass.csv")]


def write_records(d, recs):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


class Contract(unittest.TestCase):
    def test_every_kind_is_categorised_and_a_complete_record_validates(self):
        for engine, kinds in sr.KINDS.items():
            for kind, cat in {**kinds, **sr.COMMON_KINDS}.items():
                self.assertIn(cat, sr.CATEGORIES, f"{engine}.{kind}")
            rec = sr.record(engine, 10, 9, "OK", full_rows(engine, 5))
            self.assertEqual(sr.validate(rec), [])

    def test_refusals(self):
        """Missing, unknown or duplicate kinds, negative or NA-with-a-value
        bytes, an unknown run status."""
        rows = full_rows("sirius")
        for rws, msg in ((rows[1:], "missing kinds"),
                         (rows + [sr.row("bogus", 1)], "unknown kinds"),
                         (rows + [rows[0]], "duplicate kinds")):
            with self.subTest(msg), self.assertRaisesRegex(ValueError, msg):
                sr.record("sirius", 1, 1, "OK", rws)
        with self.assertRaisesRegex(ValueError, "run_status"):
            sr.record("sirius", 1, 1, "CRASHED", rows)
        for bad, msg in (([dict(rows[0], logical_b=-1)], "non-negative"),
                         ([dict(rows[0], quality="NA")], "NA with a value"),
                         ([sr.row(rows[0]["kind"])], "without a value")):
            with self.subTest(msg):
                problems = sr.validate({"engine": "sirius", "query": "query1", "run_status": "OK",
                                        "rows": bad + rows[1:]})
                self.assertTrue(any(msg in p for p in problems), problems)

    def test_checks(self):
        self.assertTrue(sr.check_result(sr.check("x", 5, 5)))
        self.assertFalse(sr.check_result(sr.check("x", 5, 6)))
        self.assertTrue(sr.check_result(sr.check("x", 100, 101, tol=0.02)))
        self.assertTrue(sr.check_result(sr.check("x", 2, 1, ">=")))
        self.assertFalse(sr.check_result(sr.check("x", 0, 1, ">=")))
        self.assertIsNone(sr.check_result(sr.check("x", None, 1)))


class PassCsv(unittest.TestCase):
    def test_last_row_of_a_query_and_the_stub(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "pass.csv")
            with open(p, "w") as f:
                f.write("query,status,seconds,result_rows_or_error\n"
                        "query9,FAIL,1.0,boom\nquery9,OK,2.500,175\nquery18,KILLED_DISK,NA,exceeded_disk_scratch\n"
                        "query5,TIMEOUT,600,exceeded\n")
            self.assertEqual(sr.pass_result(p, 9), ("OK", 2.5, 175))
            self.assertEqual(sr.pass_result(p, 18), ("KILLED_DISK", None, None))
            self.assertEqual(sr.pass_result(p, 1), ("FAIL", None, None))
            status, row = sr.pass_time(p, 18)
            self.assertEqual((status, row["quality"]), ("KILLED_DISK", "NA"))
            rec = sr.stub("sirius", 1000, 5, d, "no quent")
            self.assertEqual(rec["run_status"], "TIMEOUT")
            self.assertEqual({r["quality"] for r in rec["rows"] if r["kind"] != "pass_time"}, {"NA"})
            self.assertEqual(rec["error"], "no quent")


class Headline(unittest.TestCase):
    def test_sums_by_category(self):
        rows = {r["kind"]: r for r in full_rows("sirius", 0)}
        rows["spill_gpu_host"].update(logical_b=100, wire_b=100)
        rows["spill_gpu_disk"].update(logical_b=7, wire_b=7)
        rows["spill_host_disk"].update(logical_b=40, wire_b=40)
        h = sr.headline("sirius", list(rows.values()))
        self.assertEqual(h["off_gpu"], (107, 107, "OK"))
        self.assertEqual(h["pressure_spill"], (107, 107, "OK"))
        self.assertEqual(h["disk_spill"], (47, 47, "OK"))
        self.assertIsNone(h["design_disk"])            # no such tier: "—", not 0
        # An unmeasured part makes the sum a lower bound.
        rows = {r["kind"]: r for r in full_rows("rapids", 10)}
        rows["restore_host"] = sr.row("restore_host", seconds=0.5, source="time only")
        h = sr.headline("rapids", list(rows.values()))
        self.assertEqual(h["read_back"][2], "LOWER_BOUND")
        self.assertEqual(h["read_back"][0], 20)        # shuffle_h2d + restore_disk; restore_host bytes NA


class Merge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.res = os.path.join(self.tmp.name, "results")
        self.ev = os.path.join(self.tmp.name, "ev")

    def tearDown(self):
        self.tmp.cleanup()

    def merge(self, engine, sf, recs, settings=True):
        write_records(self.ev, recs)
        if settings:
            with open(os.path.join(self.ev, "settings.json"), "w") as f:
                json.dump({"engine": engine, "scale_factor": sf}, f)
        with redirect_stdout(io.StringIO()):
            sr.merge(engine, sf, self.ev, self.res)
        with open(os.path.join(self.res, "spill_profile.csv")) as f:
            return list(csv.DictReader(f))

    def test_upsert(self):
        """A record replaces only its query's rows (the last line of a query
        wins); other engines and SFs are kept, in order."""
        rows = self.merge("sirius", 100, [sr.record("sirius", 100, q, "OK", full_rows("sirius", q)) for q in (1, 2)])
        n = len(full_rows("sirius"))
        self.assertEqual(len(rows), 2 * n)
        rows = self.merge("sirius", 100, [sr.record("sirius", 100, 2, "FAIL", full_rows("sirius", 99)),
                                          sr.record("sirius", 100, 2, "TIMEOUT", full_rows("sirius", 98))])
        self.assertEqual(len(rows), 2 * n)
        q2 = [r for r in rows if r["query"] == "query2"]
        self.assertEqual({r["run_status"] for r in q2}, {"TIMEOUT"})
        self.assertEqual([r for r in rows if r["query"] == "query1"][0]["logical_b"], "1")
        self.assertEqual(rows[0]["kind"], "scan")      # CSV order: KINDS order within a query
        with open(os.path.join(self.res, "spill_settings.json")) as f:
            self.assertEqual(list(json.load(f)["sf100"]), ["sirius"])
        self.merge("rapids", 100, [sr.record("rapids", 100, 9, "OK", full_rows("rapids"))])
        rows = self.merge("sirius", 1000, [sr.record("sirius", 1000, 9, "OK", full_rows("sirius"))])
        order = [(r["engine"], r["scale_factor"]) for r in rows]
        self.assertEqual(sorted(set(order), key=order.index),
                         [("sirius", "100"), ("sirius", "1000"), ("rapids", "100")])

    def test_refusals(self):
        with self.assertRaises(SystemExit):        # a record of another SF
            self.merge("sirius", 100, [sr.record("sirius", 10, 9, "OK", full_rows("sirius"))])
        bad = sr.record("sirius", 100, 9, "OK", full_rows("sirius"))
        bad["rows"] = bad["rows"][1:]
        with self.assertRaises(SystemExit):        # a contract violation
            self.merge("sirius", 100, [bad])
        os.makedirs(self.res, exist_ok=True)
        with open(os.path.join(self.res, "spill_profile.csv"), "w") as f:
            f.write("engine,query\n")
        with self.assertRaises(SystemExit):        # a header mismatch
            self.merge("sirius", 100, [sr.record("sirius", 100, 9, "OK", full_rows("sirius"))])
        self.assertFalse(os.path.exists(os.path.join(self.res, "spill_profile.csv.tmp")))


class CheckDirs(unittest.TestCase):
    def test_check_fails_on_a_failed_identity_and_without_spill(self):
        with tempfile.TemporaryDirectory() as d:
            good = sr.record("sirius", 1, 9, "OK", full_rows("sirius", 0), [sr.check("eq", 3, 3)])
            write_records(d, [good])
            with redirect_stdout(io.StringIO()):
                self.assertTrue(sr.check_dirs([d], expect_spill=False))
                self.assertFalse(sr.check_dirs([d], expect_spill=True))     # nothing spilled
            write_records(d, [sr.record("sirius", 1, 9, "OK", full_rows("sirius", 5), [sr.check("eq", 3, 4)])])
            with redirect_stdout(io.StringIO()):
                self.assertFalse(sr.check_dirs([d], expect_spill=True))


if __name__ == "__main__":
    unittest.main()
