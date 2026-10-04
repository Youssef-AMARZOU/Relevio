"""Convert Instacart market-basket CSVs into OTTO-shaped JSONL for the demo.

Each Instacart order becomes one session; its products, ordered by
``add_to_cart_order``, become the event sequence. Instacart only records
purchases, so event types are assigned with the same 80/10/10
clicks/carts/orders mix used by ``make_sample_jsonl.py``, seeded
deterministically per order. Timestamps are synthesized from the order's
``order_dow``/``order_hour_of_day`` across 2024 so the temporal split and
replay analysis see a realistic spread of dates.

The trailing run of order ids is dropped: the fact file may be cut
mid-order, and only a complete order can be a session.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

TYPES = ["clicks"] * 8 + ["carts"] * 1 + ["orders"] * 1
YEAR = 2024


def load_orders(path: Path) -> dict[int, tuple[int, int]]:
    """Map order_id -> (order_dow, order_hour_of_day)."""
    meta: dict[int, tuple[int, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            meta[int(row["order_id"])] = (int(row["order_dow"]), int(row["order_hour_of_day"]))
    return meta


def group_fact(path: Path, max_rows: int = 0) -> list[tuple[int, list[tuple[int, int]]]]:
    """Group fact rows into (order_id, [(add_to_cart_order, product_id), ...]).

    Rows must be contiguous per order id (Instacart ships them grouped).
    Reading stops after ``max_rows`` rows (0 = all); the trailing group is
    dropped because it may be cut mid-order.
    """
    groups: list[tuple[int, list[tuple[int, int]]]] = []
    current_id: int | None = None
    current: list[tuple[int, int]] = []
    seen: set[int] = set()
    rows_read = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if max_rows and rows_read >= max_rows:
                break
            rows_read += 1
            order_id = int(row["order_id"])
            if order_id != current_id:
                if current_id is not None:
                    if current_id in seen:
                        raise ValueError(f"order {current_id} reappears after another group; file is not contiguous")
                    seen.add(current_id)
                    groups.append((current_id, current))
                current_id = order_id
                current = []
            current.append((int(row["add_to_cart_order"]), int(row["product_id"])))
    if not groups:
        raise ValueError(f"no grouped orders found in {path}")
    return groups  # pending ``current`` is the trailing (possibly partial) run


def _dates_by_dow() -> dict[int, list[datetime]]:
    by_dow: dict[int, list[datetime]] = {i: [] for i in range(7)}
    day = datetime(YEAR, 1, 1, tzinfo=timezone.utc)
    while day.year == YEAR:
        by_dow[day.weekday()].append(day)  # Monday=0
        day += timedelta(days=1)
    return by_dow


def build_sessions(
    groups: list[tuple[int, list[tuple[int, int]]]],
    meta: dict[int, tuple[int, int]],
) -> tuple[list[dict], int]:
    """Build OTTO events-schema sessions sorted by first timestamp."""
    by_dow = _dates_by_dow()
    sessions: list[tuple[datetime, dict]] = []
    skipped = 0
    for order_id, rows in groups:
        if order_id not in meta:
            skipped += 1
            continue
        order_dow, order_hour = meta[order_id]
        # Instacart dow: 0 = Sunday; Python weekday: Monday = 0.
        candidates = by_dow[(order_dow + 1) % 7]
        start = candidates[(order_id // 7) % len(candidates)].replace(
            hour=order_hour, minute=order_id % 60, second=(order_id // 60) % 60
        )
        rng = random.Random(order_id)  # noqa: S311 - deterministic demo data, not cryptographic
        rows.sort()
        events = []
        ts = int(start.timestamp() * 1000)
        for pos, (_order, product_id) in enumerate(rows):
            if pos:
                ts += 5_000 + (order_id * 131 + pos * 17) % 120_000
            events.append({"aid": product_id, "ts": ts, "type": rng.choice(TYPES)})
        sessions.append((start, {"session": order_id, "events": events}))
    sessions.sort(key=lambda item: item[0])
    return [record for _start, record in sessions], skipped


def convert(fact_path: Path, orders_path: Path, out_path: Path, max_rows: int = 1_000_000) -> dict[str, int]:
    meta = load_orders(orders_path)
    groups = group_fact(fact_path, max_rows=max_rows)
    sessions, skipped = build_sessions(groups, meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in sessions:
            handle.write(json.dumps(record) + "\n")
    stats = {
        "sessions": len(sessions),
        "events": sum(len(record["events"]) for record in sessions),
        "trailing_group_dropped": 1,
        "missing_order_metadata": skipped,
    }
    print(
        f"wrote {stats['sessions']} sessions / {stats['events']} events -> {out_path} "
        f"(dropped trailing group, {skipped} orders without metadata)"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact", default="data/demo/instacart/order_products__train.csv")
    parser.add_argument("--orders", default="data/demo/instacart/orders.csv")
    parser.add_argument("--out", default="data/demo/otto_train_1m.jsonl")
    parser.add_argument("--max-rows", type=int, default=1_000_000, help="0 reads the whole fact file")
    args = parser.parse_args()
    convert(Path(args.fact), Path(args.orders), Path(args.out), max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
