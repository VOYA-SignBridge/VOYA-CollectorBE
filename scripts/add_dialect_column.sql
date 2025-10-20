-- Migration: Add dialect column to support sign language dialect variations
-- Run this migration if you're using a database instead of CSV files

-- For future database migration (currently using CSV files)
-- ALTER TABLE samples ADD COLUMN dialect VARCHAR(128) NULL;
-- CREATE INDEX idx_samples_dialect ON samples(dialect);

-- Note: This project currently uses CSV files (dataset/samples.csv)
-- The dialect field has been added to the CSV structure via storage_utils.py
-- If migrating to a proper database, uncomment the SQL above

-- To verify dialect support in CSV:
-- Check that dataset/samples.csv now includes a "dialect" column
-- Upload test data with FormData dialect="Bắc" or JSON {"dialect": "Nam"}