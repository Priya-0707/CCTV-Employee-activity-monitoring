"""
db.py - all database code lives here (SQLite = a real SQL database stored in one file).

Tables
  employees     one row per registered employee
  face_samples  one or more face "prints" (128 numbers each) per employee
"""

import os
import sqlite3
from datetime import datetime

import numpy as np

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "attendance.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row            # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")  # needed for ON DELETE CASCADE
    return conn


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_code    TEXT NOT NULL UNIQUE,   -- the company's employee ID, e.g. EMP001
                name        TEXT NOT NULL,
                department  TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS face_samples (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id  INTEGER NOT NULL
                             REFERENCES employees(id) ON DELETE CASCADE,
                embedding    BLOB NOT NULL,         -- 128 float32 numbers
                image_path   TEXT,
                created_at   TEXT NOT NULL
            );
            """
        )


def _now():
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ employees
def add_employee(emp_code, name, department):
    """Returns the new employee's database id. Raises sqlite3.IntegrityError
    if the emp_code already exists."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO employees (emp_code, name, department, created_at) VALUES (?, ?, ?, ?)",
            (emp_code, name, department, _now()),
        )
        return cur.lastrowid


def add_face_sample(employee_id, embedding, image_path):
    blob = np.asarray(embedding, dtype=np.float32).tobytes()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO face_samples (employee_id, embedding, image_path, created_at) "
            "VALUES (?, ?, ?, ?)",
            (employee_id, blob, image_path, _now()),
        )


def list_employees():
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT e.id, e.emp_code, e.name, e.department, e.created_at,
                   COUNT(f.id) AS photos
            FROM employees e
            LEFT JOIN face_samples f ON f.employee_id = e.id
            GROUP BY e.id
            ORDER BY e.created_at DESC
            """
        ).fetchall()


def delete_employee(employee_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM employees WHERE id = ?", (employee_id,))  # samples cascade


def load_gallery():
    """All stored face prints, ready for matching (used in Step 2).
    Returns a list of (employee_id, emp_code, name, department, embedding_array)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.emp_code, e.name, e.department, f.embedding
            FROM face_samples f JOIN employees e ON e.id = f.employee_id
            """
        ).fetchall()
    return [(r["id"], r["emp_code"], r["name"], r["department"],
             np.frombuffer(r["embedding"], dtype=np.float32))
            for r in rows]
