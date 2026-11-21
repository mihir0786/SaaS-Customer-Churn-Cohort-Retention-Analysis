"""
Load the raw CSVs into DuckDB and run every SQL file in sql/ in order.
Each table a SQL file creates is also exported to outputs/tables/<table>.csv.

Usage:  python -m src.run_sql
"""
import re
import time

import duckdb

from src.config import DB_PATH, RAW, SQL_DIR, TABLES


def load_raw(con):
    for name in ("customers", "monthly_usage", "cancellations"):
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT * FROM read_csv_auto('{(RAW / f"{name}.csv").as_posix()}', header = true)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  loaded {name:<15} {n:>9,} rows")


def run_sql_files(con):
    created = []
    for path in sorted(SQL_DIR.glob("*.sql")):
        t0 = time.time()
        con.execute(path.read_text())
        tables = re.findall(r"CREATE OR REPLACE TABLE\s+(\w+)", path.read_text(), flags=re.I)
        created += tables
        print(f"  ran {path.name:<32} -> {', '.join(tables)}  ({time.time() - t0:.1f}s)")
    return created


def export(con, tables):
    for t in tables:
        if t == "customer_months":          # large intermediate table, not needed as CSV
            continue
        out = TABLES / f"{t}.csv"
        con.execute(f"COPY {t} TO '{out.as_posix()}' (HEADER, DELIMITER ',')")
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  exported {out.name:<34} {n:>8,} rows")


def main():
    DB_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(DB_PATH))
    print("Loading raw data")
    load_raw(con)
    print("Running SQL")
    tables = run_sql_files(con)
    print("Exporting tables")
    export(con, tables)
    con.close()


if __name__ == "__main__":
    main()
