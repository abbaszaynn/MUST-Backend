import sqlite3

DB_NAME = "hatespeech.db"

try:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Check schema
    print("--- Schema ---")
    c.execute("PRAGMA table_info(logs)")
    columns = c.fetchall()
    for col in columns:
        print(col)
        
    # Check data count
    print("\n--- Data Count ---")
    c.execute("SELECT count(*) FROM logs")
    count = c.fetchone()[0]
    print(f"Total logs: {count}")
    
    # Check last 5 rows
    print("\n--- Last 5 Rows ---")
    c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 5")
    rows = c.fetchall()
    for row in rows:
        print(row)
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
