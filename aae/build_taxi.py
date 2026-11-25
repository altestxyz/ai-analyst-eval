import duckdb

from aae.config import DB_DIR, TAXI_DIR

DB_DIR.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(str(DB_DIR / "nyc_taxi.duckdb"))
con.execute(f"CREATE OR REPLACE TABLE trips AS SELECT * FROM read_parquet('{TAXI_DIR}/yellow_tripdata_2025-*.parquet', union_by_name=true)")
con.execute(f"CREATE OR REPLACE TABLE zones AS SELECT * FROM read_csv_auto('{TAXI_DIR}/taxi_zone_lookup.csv')")
print(con.execute("SELECT count(*) FROM trips").fetchone(), con.execute("SELECT count(*) FROM zones").fetchone())
print([r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='trips'").fetchall()])
