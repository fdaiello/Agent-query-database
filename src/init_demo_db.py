import sqlite3

def initialize_demo_db(db_path="example.db"):
    """Create demo tables and populate with small fake data."""
    schema_sql = '''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL
    );
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        price REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        order_date TEXT NOT NULL,
        FOREIGN KEY(customer_id) REFERENCES customers(id),
        FOREIGN KEY(product_id) REFERENCES products(id)
    );
    '''
    data_sql = [
        # Customers
        ("INSERT INTO customers (name, email) VALUES (?, ?)", ("Alice Smith", "alice@example.com")),
        ("INSERT INTO customers (name, email) VALUES (?, ?)", ("Bob Jones", "bob@example.com")),
        # Products
        ("INSERT INTO products (name, price) VALUES (?, ?)", ("Widget", 19.99)),
        ("INSERT INTO products (name, price) VALUES (?, ?)", ("Gadget", 29.99)),
        ("INSERT INTO products (name, price) VALUES (?, ?)", ("Thingamajig", 9.99)),
        # Orders
        ("INSERT INTO orders (customer_id, product_id, quantity, order_date) VALUES (?, ?, ?, ?)", (1, 1, 2, "2025-08-01")),
        ("INSERT INTO orders (customer_id, product_id, quantity, order_date) VALUES (?, ?, ?, ?)", (1, 3, 1, "2025-08-02")),
        ("INSERT INTO orders (customer_id, product_id, quantity, order_date) VALUES (?, ?, ?, ?)", (2, 2, 3, "2025-08-03")),
    ]
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for stmt in schema_sql.strip().split(';'):
        if stmt.strip():
            cur.execute(stmt)
    for sql, params in data_sql:
        cur.execute(sql, params)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_demo_db()
