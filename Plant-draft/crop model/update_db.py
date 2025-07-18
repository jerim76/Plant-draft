import sqlite3

# Connect to the database
conn = sqlite3.connect('agritech.db')
cursor = conn.cursor()

# Add tags column to Post table
cursor.execute("ALTER TABLE post ADD COLUMN tags TEXT")
conn.commit()

# Verify the change
cursor.execute("PRAGMA table_info(post)")
columns = [info[1] for info in cursor.fetchall()]
print("Post table columns:", columns)

conn.close()