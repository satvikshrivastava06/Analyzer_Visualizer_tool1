import duckdb
import os
import toml

print("Loading secrets...")
try:
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
        md_token = secrets["motherduck"]["token"]
        print("Token loaded successfully. Length:", len(md_token))
except Exception as e:
    print("Error loading secrets:", e)

os.environ["MOTHERDUCK_TOKEN"] = md_token

print("Attempting to connect to MotherDuck...")
try:
    conn = duckdb.connect('md:')
    print("Connection successful!")
    print(conn.execute("SELECT 1").df())
except Exception as e:
    print("Connection failed:", e)
