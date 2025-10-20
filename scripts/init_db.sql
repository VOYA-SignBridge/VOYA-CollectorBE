-- run manually if desired
CREATE TABLE IF NOT EXISTS labels (
  id SERIAL PRIMARY KEY,
  class_idx INTEGER NOT NULL,
  label_name TEXT NOT NULL,
  folder_name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS samples (
  id SERIAL PRIMARY KEY,
  label_id INTEGER,
  file_path TEXT NOT NULL,
  "user" TEXT,
  session_id TEXT,
  frames INTEGER,
  duration TEXT,
  meta JSON,
  created_at TIMESTAMP DEFAULT now()
);
