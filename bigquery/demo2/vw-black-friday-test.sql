-- Creating a View from black-friday-test

CREATE VIEW `bt-int-ml-specialization.demo2.vw-black-friday-test` AS
WITH transformation_cols AS (
  SELECT
    *,
    -- 1. Filling the NULL values of product category 2 and 3 with valid values
    COALESCE(Product_Category_2, 1) AS Product_Category_2_new,
    COALESCE(Product_Category_3, 2) AS Product_Category_3_new,
    
    -- 2. Transforming M and F into 0 and 1 (Gender column)
    CASE
      WHEN gender = 'M' THEN 0
      WHEN gender = 'F' THEN 1
    END AS Gender_New,

    -- Update new columns with One-Hot Encoded values
    -- For age groups
    IF(Age = '0-17', 1, 0) AS Age_0_17,
    IF(Age = '18-25', 1, 0) AS Age_18_25,
    IF(Age = '26-35', 1, 0) AS Age_26_35,
    IF(Age = '36-45', 1, 0) AS Age_36_45,
    IF(Age = '46-50', 1, 0) AS Age_46_50,
    IF(Age = '51-55', 1, 0) AS Age_51_55,
    IF(Age = '55+', 1, 0) AS Age_55_plus,

    -- For city categories
    IF(City_Category = 'A', 1, 0) AS City_Category_A,
    IF(City_Category = 'B', 1, 0) AS City_Category_B,
    IF(City_Category = 'C', 1, 0) AS City_Category_C,

    -- For stay in city
    IF(Stay_In_Current_City_Years = '0', 1, 0) AS Stay_0_years,
    IF(Stay_In_Current_City_Years = '1', 1, 0) AS Stay_1_years,
    IF(Stay_In_Current_City_Years = '2', 1, 0) AS Stay_2_years,
    IF(Stay_In_Current_City_Years = '3', 1, 0) AS Stay_3_years,
    IF(Stay_In_Current_City_Years = '4+', 1, 0) AS Stay_4_plus_years
  FROM
    `bt-int-ml-specialization.demo2.black-friday-test`
)

-- Selecting only the columns needed for the final result
SELECT

    User_ID,
    Product_ID,
    Occupation,
    Marital_Status,
    Gender_New AS Gender,

    Product_Category_1,
    Product_Category_2_new AS Product_Category_2,
    Product_Category_3_new AS Product_Category_3,
  
    Age_0_17,
    Age_18_25,
    Age_26_35,
    Age_36_45,
    Age_46_50,
    Age_51_55,
    Age_55_plus,

    City_Category_A,
    City_Category_B,
    City_Category_C,
    
    Stay_0_years,
    Stay_1_years,
    Stay_2_years,
    Stay_3_years,
    Stay_4_plus_years

FROM
transformation_cols;
