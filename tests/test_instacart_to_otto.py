"""Tests for scripts/instacart_to_otto.py (stdlib-only, no pandas)."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import instacart_to_otto as conv  # noqa: E402

FACT_HEADER = ["order_id", "product_id", "add_to_cart_order", "reordered"]
ORDERS_HEADER = [
    "order_id",
    "user_id",
    "eval_set",
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
]


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class ConvertTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.fact = base / "fact.csv"
        self.orders = base / "orders.csv"
        self.out = base / "out.jsonl"
        _write_csv(
            self.fact,
            FACT_HEADER,
            [
                [100, 11, 1, 1],
                [100, 12, 2, 0],
                [100, 13, 3, 0],
                [100, 14, 4, 1],
                [200, 21, 1, 0],
                [200, 22, 2, 1],
                [200, 23, 3, 0],
                [300, 31, 1, 0],  # trailing group -> dropped (possible mid-order cut)
                [300, 32, 2, 1],
            ],
        )
        _write_csv(
            self.orders,
            ORDERS_HEADER,
            [
                [100, 1, "train", 1, 0, 8, ""],
                [200, 2, "train", 1, 3, 17, 15],
                [300, 3, "train", 1, 5, 21, 7],
            ],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_stats_and_schema(self) -> None:
        stats = conv.convert(self.fact, self.orders, self.out)
        self.assertEqual(stats["sessions"], 2)
        self.assertEqual(stats["events"], 7)
        self.assertEqual(stats["trailing_group_dropped"], 1)
        self.assertEqual(stats["missing_order_metadata"], 0)

        records = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["session"] for r in records], [100, 200])
        for record in records:
            self.assertEqual(set(record), {"session", "events"})
            ts_prev = None
            for event in record["events"]:
                self.assertEqual(set(event), {"aid", "ts", "type"})
                self.assertIn(event["type"], {"clicks", "carts", "orders"})
                self.assertIsInstance(event["aid"], int)
                if ts_prev is not None:
                    self.assertGreater(event["ts"], ts_prev)
                ts_prev = event["ts"]

    def test_sorted_by_first_ts(self) -> None:
        conv.convert(self.fact, self.orders, self.out)
        records = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        firsts = [r["events"][0]["ts"] for r in records]
        self.assertEqual(firsts, sorted(firsts))

    def test_deterministic(self) -> None:
        other = Path(self.tmp.name) / "out2.jsonl"
        conv.convert(self.fact, self.orders, self.out)
        conv.convert(self.fact, self.orders, other)
        self.assertEqual(self.out.read_bytes(), other.read_bytes())

    def test_missing_order_metadata_skipped(self) -> None:
        # Group 999 is followed by 100, so it is complete but has no orders.csv row.
        # Group 300 is trailing and dropped by the mid-order-cut rule.
        _write_csv(
            self.fact,
            FACT_HEADER,
            [
                [999, 91, 1, 0],
                [100, 11, 1, 0],
                [300, 31, 1, 0],
            ],
        )
        stats = conv.convert(self.fact, self.orders, self.out)
        self.assertEqual(stats["sessions"], 1)
        self.assertEqual(stats["missing_order_metadata"], 1)

    def test_non_contiguous_orders_raise(self) -> None:
        # The duplicated 100 run must be flushed by a following group to be detected.
        _write_csv(
            self.fact,
            FACT_HEADER,
            [
                [100, 11, 1, 0],
                [200, 21, 1, 0],
                [100, 12, 2, 0],
                [300, 31, 1, 0],
            ],
        )
        with self.assertRaises(ValueError):
            conv.convert(self.fact, self.orders, self.out)

    def test_single_event_order(self) -> None:
        # Trailing group (300) is dropped; 100 and 200 remain as one-event sessions.
        _write_csv(self.fact, FACT_HEADER, [[100, 11, 1, 0], [200, 21, 1, 0], [300, 31, 1, 0]])
        stats = conv.convert(self.fact, self.orders, self.out)
        records = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(stats["sessions"], 2)
        self.assertEqual([len(r["events"]) for r in records], [1, 1])

    def test_max_rows_cuts_and_drops_partial_group(self) -> None:
        # max_rows=3 keeps order 100 complete and cuts 200 mid-order -> dropped.
        _write_csv(
            self.fact,
            FACT_HEADER,
            [
                [100, 11, 1, 0],
                [100, 12, 2, 0],
                [200, 21, 1, 0],
                [200, 22, 2, 0],
            ],
        )
        stats = conv.convert(self.fact, self.orders, self.out, max_rows=3)
        self.assertEqual(stats["sessions"], 1)
        self.assertEqual(stats["events"], 2)
        records = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[0]["session"], 100)


if __name__ == "__main__":
    unittest.main()
