-- Show the training data set

SELECT * FROM `bt-int-ml-specialization.demo2.black-friday-train` LIMIT 1000;


-- Analysis of the independent Variables

SELECT COUNT(DISTINCT User_ID) AS TotalDistinctUsers FROM `bt-int-ml-specialization.demo2.black-friday-train`;
-- 5891 unique users

SELECT COUNT(DISTINCT Product_ID) AS TotalDistinctUsers FROM `bt-int-ml-specialization.demo2.black-friday-train`;
-- 3631 unique products

SELECT DISTINCT Gender FROM `bt-int-ml-specialization.demo2.black-friday-train`;
-- available genders = M and F

SELECT DISTINCT Age FROM `bt-int-ml-specialization.demo2.black-friday-train`;
-- age ordinal: 0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+

SELECT DISTINCT occupation FROM `bt-int-ml-specialization.demo2.black-friday-train`;
-- 0 to 20 categories (21)

SELECT DISTINCT City_Category FROM `bt-int-ml-specialization.demo2.black-friday-train`;
-- Categories: A, B, C

SELECT DISTINCT Stay_In_Current_City_Years FROM `bt-int-ml-specialization.demo2.black-friday-train`;
-- 0, 1, 2, 3, 4+

SELECT DISTINCT Marital_Status FROM `bt-int-ml-specialization.demo2.black-friday-train`;
SELECT Age, Marital_Status FROM `bt-int-ml-specialization.demo2.black-friday-train`
WHERE Marital_Status = 1
AND Age = '0-17';
-- 0 = ledig, 1 = verheiratet

SELECT DISTINCT Product_Category_1 FROM `bt-int-ml-specialization.demo2.black-friday-train`
ORDER BY Product_Category_1;
-- 1 to 20 (20)

SELECT DISTINCT Product_Category_2 FROM `bt-int-ml-specialization.demo2.black-friday-train`
ORDER BY Product_Category_2;
-- 2 to 18 (18)
-- 1 = null

SELECT DISTINCT Product_Category_3 FROM `bt-int-ml-specialization.demo2.black-friday-train`
ORDER BY Product_Category_3;
-- 3 to 18 (16)
-- 1 = null
-- 2 = not existing

-- Checking for NULL values in the data set
SELECT COUNT(*) AS NullCount
FROM `bt-int-ml-specialization.demo2.black-friday-train` 
WHERE User_ID IS NULL
OR Product_ID IS NULL
OR Gender IS NULL
OR Age IS NULL
OR Occupation IS NULL
OR City_Category IS NULL
OR Stay_In_Current_City_Years IS NULL
OR Marital_Status IS NULL
OR Purchase IS NULL
OR Product_Category_1 IS NULL;
-- clean columns

SELECT COUNT(*)
FROM `bt-int-ml-specialization.demo2.black-friday-train`
WHERE Product_Category_2 IS NULL
OR Product_Category_3 IS NULL;
-- 383247 NULL values in product category 2 and 3

SELECT DISTINCT Product_Category_2 FROM `bt-int-ml-specialization.demo2.black-friday-train`
ORDER BY Product_Category_2;
-- Missing Value = 1

SELECT DISTINCT Product_Category_3 FROM `bt-int-ml-specialization.demo2.black-friday-train`
ORDER BY Product_Category_3;
-- Missing Value = 2


-- Analysis of the dependent Variable
SELECT MIN(Purchase) as min_purchase,
MAX(Purchase) as max_purchase,
AVG(Purchase) as mean_purchase,
APPROX_QUANTILES(Purchase, 2)[OFFSET(1)] AS median_value
FROM `bt-int-ml-specialization.demo2.black-friday-train`;
-- min = 12.00
-- max = 23961.00
-- mean = 9263.97
-- median = 8020.00

-- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- --

-- Show the training data set

SELECT * FROM `bt-int-ml-specialization.demo2.black-friday-test` LIMIT 1000;


-- Analysis of the independent Variables

SELECT COUNT(DISTINCT User_ID) AS TotalDistinctUsers FROM `bt-int-ml-specialization.demo2.black-friday-test`;
-- 5891 unique users

SELECT COUNT(DISTINCT Product_ID) AS TotalDistinctUsers FROM `bt-int-ml-specialization.demo2.black-friday-test`;
-- 3491 unique products

SELECT DISTINCT Gender FROM `bt-int-ml-specialization.demo2.black-friday-test`;
-- available genders = M and F

SELECT DISTINCT Age FROM `bt-int-ml-specialization.demo2.black-friday-test`;
-- age ordinal: 0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+

SELECT DISTINCT occupation FROM `bt-int-ml-specialization.demo2.black-friday-test`;
-- 0 to 20 categories (21) 

SELECT DISTINCT City_Category FROM `bt-int-ml-specialization.demo2.black-friday-test`;
-- Categories: A, B, C

SELECT DISTINCT Stay_In_Current_City_Years FROM `bt-int-ml-specialization.demo2.black-friday-test`;
-- 0, 1, 2, 3, 4+

SELECT DISTINCT Marital_Status FROM `bt-int-ml-specialization.demo2.black-friday-test`;
SELECT Age, Marital_Status FROM `bt-int-ml-specialization.demo2.black-friday-test`
WHERE Marital_Status = 1
AND Age = '0-17';
-- 0 = ledig, 1 = verheiratet

SELECT DISTINCT Product_Category_1 FROM `bt-int-ml-specialization.demo2.black-friday-test`
ORDER BY Product_Category_1;
-- 1 to 18 (18)

SELECT DISTINCT Product_Category_2 FROM `bt-int-ml-specialization.demo2.black-friday-test`
ORDER BY Product_Category_2;
-- 2 to 18 (18)
-- 1 = null

SELECT DISTINCT Product_Category_3 FROM `bt-int-ml-specialization.demo2.black-friday-test`
ORDER BY Product_Category_3;
-- 3 to 18 (16)
-- 1 = null
-- 2 = not existing

-- Checking for NULL values in the data set
SELECT COUNT(*) AS NullCount
FROM `bt-int-ml-specialization.demo2.black-friday-test` 
WHERE User_ID IS NULL
OR Product_ID IS NULL
OR Gender IS NULL
OR Age IS NULL
OR Occupation IS NULL
OR City_Category IS NULL
OR Stay_In_Current_City_Years IS NULL
OR Marital_Status IS NULL
OR Product_Category_1 IS NULL;

SELECT COUNT(*)
FROM `bt-int-ml-specialization.demo2.black-friday-test`
WHERE Product_Category_2 IS NULL
OR Product_Category_3 IS NULL;
-- 162562 NULL values in product category 2 and 3

SELECT DISTINCT Product_Category_2 FROM `bt-int-ml-specialization.demo2.black-friday-test`
ORDER BY Product_Category_2;
-- Missing Value = 1

SELECT DISTINCT Product_Category_3 FROM `bt-int-ml-specialization.demo2.black-friday-test`
ORDER BY Product_Category_3;
-- Missing Value = 2
