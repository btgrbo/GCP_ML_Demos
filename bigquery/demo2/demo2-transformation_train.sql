-- Create a new table to avoid data corruption
/**
CREATE TABLE `bt-int-ml-specialization.demo2.black-friday-train-new`
AS
SELECT *
FROM `bt-int-ml-specialization.demo2.black-friday-train`;
**/


-- 1. Filling the NULL values of product category 2 and 3 with valid values
UPDATE `bt-int-ml-specialization.demo2.black-friday-train-new`
SET
  Product_Category_2 = COALESCE(Product_Category_2, 1),
  Product_Category_3 = COALESCE(Product_Category_3, 2)
WHERE
  Product_Category_2 IS NULL OR Product_Category_3 IS NULL
;

-- Checking success of operation
SELECT Count(*) FROM `bt-int-ml-specialization.demo2.black-friday-train-new`
WHERE Product_Category_2 IS NULL
OR Product_Category_3 IS NULL;


-- 2. Transforming M and F into 0 and 1
--    M = 0 & F = 1

-- Add a new gender column with the type int
ALTER TABLE `bt-int-ml-specialization.demo2.black-friday-train-new`
ADD COLUMN Gender_New INT64;

-- Update the temporary column based on '0' and '1' values
UPDATE `bt-int-ml-specialization.demo2.black-friday-train-new`
SET Gender_New = 
  CASE
    WHEN gender = 'M' THEN 0
    WHEN gender = 'F' THEN 1
  END
WHERE Gender IN ('M', 'F');

-- Drop the original column
ALTER TABLE `bt-int-ml-specialization.demo2.black-friday-train-new`
DROP COLUMN Gender;


-- 3. Performing OHE on nominal variables
 
-- Selection of columns to perform on
SELECT
  column_name,
  data_type
FROM
  `bt-int-ml-specialization.demo2.INFORMATION_SCHEMA.COLUMNS`
WHERE
table_name = 'black-friday-train-new';


-- Columns to perform OHE on:
-- Age, City_Category, Stay_IN_Current_City_Years

-- Adding new columns to the table
ALTER TABLE `bt-int-ml-specialization.demo2.black-friday-train-new`
ADD COLUMN Age_0_17 INT64,
ADD COLUMN Age_18_25 INT64,
ADD COLUMN Age_26_35 INT64,
ADD COLUMN Age_36_45 INT64,
ADD COLUMN Age_46_50 INT64,
ADD COLUMN Age_51_55 INT64,
ADD COLUMN Age_55_plus INT64,
ADD COLUMN City_Category_A INT64,
ADD COLUMN City_Category_B INT64,
ADD COLUMN City_Category_C INT64,
ADD COLUMN Stay_0_years INT64,
ADD COLUMN Stay_1_years INT64,
ADD COLUMN Stay_2_years INT64,
ADD COLUMN Stay_3_years INT64,
ADD COLUMN Stay_4_plus_years INT64;

-- Update new columns with One-Hot Encoded values
UPDATE `bt-int-ml-specialization.demo2.black-friday-train-new`
SET
  Age_0_17 = IF(Age = '0-17', 1, 0),
  Age_18_25 = IF(Age = '18-25', 1, 0),
  Age_26_35 = IF(Age = '26-35', 1, 0),
  Age_36_45 = IF(Age = '36-45', 1, 0),
  Age_46_50 = IF(Age = '46-50', 1, 0),
  Age_51_55 = IF(Age = '51-55', 1, 0),
  Age_55_plus = IF(Age = '55+', 1, 0),
  City_Category_A = IF(City_Category = 'A', 1, 0),
  City_Category_B = IF(City_Category = 'B', 1, 0),
  City_Category_C = IF(City_Category = 'C', 1, 0),
  Stay_0_years = IF(Stay_In_Current_City_Years = '0', 1, 0),
  Stay_1_years = IF(Stay_In_Current_City_Years = '1', 1, 0),
  Stay_2_years = IF(Stay_In_Current_City_Years = '2', 1, 0),
  Stay_3_years = IF(Stay_In_Current_City_Years = '3', 1, 0),
  Stay_4_plus_years = IF(Stay_In_Current_City_Years = '4+', 1, 0)
WHERE true;

-- Drop original columns
ALTER TABLE `bt-int-ml-specialization.demo2.black-friday-train-new`
DROP COLUMN Age,
DROP COLUMN City_Category,
DROP COLUMN Stay_In_Current_City_Years;

-- Checking whether transformations were successful
SELECT
  column_name,
  data_type
FROM
  `bt-int-ml-specialization.demo2.INFORMATION_SCHEMA.COLUMNS`
WHERE
table_name = 'black-friday-train-new';


-- Drop User_ID and Product_ID for prediction
ALTER TABLE `bt-int-ml-specialization.demo2.black-friday-train-new`
DROP COLUMN User_ID,
DROP COLUMN Product_ID;

-- Show columns
SELECT * FROM `bt-int-ml-specialization.demo2.black-friday-train-new`;

