-- Migration: Add dialect support to samples table
-- Date: 2025-10-20
-- Description: Add dialect column to track sign language dialects (Bắc, Trung, Nam, etc.)

-- Add dialect column to samples table
ALTER TABLE samples ADD COLUMN dialect VARCHAR(128);

-- Create index for faster queries by dialect
CREATE INDEX idx_samples_dialect ON samples(dialect);

-- Optional: Update existing records to 'unknown' if you want explicit values
-- UPDATE samples SET dialect = 'unknown' WHERE dialect IS NULL;

-- Add comment for documentation
COMMENT ON COLUMN samples.dialect IS 'Sign language dialect (e.g., Bắc, Trung, Nam)';