DECLARE dist_payment_type, dist_day_name ARRAY<STRING>;

DECLARE
     dist_start_minute,
     dist_end_minute,
     dist_start_hour,
     dist_end_hour,
     dist_start_day,
     dist_end_day,
     dist_start_month,
     dist_end_month,
     dist_start_year,
     dist_end_year ARRAY<Int64>;

SET (dist_payment_type,
     dist_day_name,
     dist_start_minute,
     dist_end_minute,
     dist_start_hour,
     dist_end_hour,
     dist_start_day,
     dist_end_day,
     dist_start_month,
     dist_end_month,
     dist_start_year,
     dist_end_year
     ) = (SELECT AS STRUCT ARRAY_AGG(DISTINCT payment_type),
                           ARRAY_AGG(DISTINCT day_name),
                           ARRAY_AGG(DISTINCT trip_start_minute),
                           ARRAY_AGG(DISTINCT trip_end_minute),
                           ARRAY_AGG(DISTINCT trip_start_hour),
                           ARRAY_AGG(DISTINCT trip_end_hour),
                           ARRAY_AGG(DISTINCT trip_start_day),
                           ARRAY_AGG(DISTINCT trip_end_day),
                           ARRAY_AGG(DISTINCT trip_start_month),
                           ARRAY_AGG(DISTINCT trip_end_month),
                           ARRAY_AGG(DISTINCT trip_start_year),
                           ARRAY_AGG(DISTINCT trip_end_year)
 FROM `bt-int-ml-specialization.demo1.taxi_trips_preproc`);


EXECUTE IMMEDIATE '''
CREATE TEMP TABLE result AS  -- added line
SELECT unique_key,
  ROW_NUMBER() OVER (ORDER BY RAND()) as row_num,
  trip_seconds_standardized,
  trip_miles_standardized,
  tips_standardized,
  extras_standardized,
  trip_total_standardized,
  label,
''' ||
 (
  SELECT STRING_AGG("COUNTIF(payment_type = '" || payment_type || "') AS payment_type_" || payment_type ORDER BY payment_type)
  FROM UNNEST(dist_payment_type) AS payment_type
) ||
(
  SELECT ', ' || STRING_AGG("COUNTIF(day_name = '" || day_name || "') AS day_name_" || day_name ORDER BY day_name)
  FROM UNNEST(dist_day_name) AS day_name
)  ||
(
  SELECT ', ' || STRING_AGG("COUNTIF(trip_start_minute = " || trip_start_minute || ") AS trip_start_minute_" || trip_start_minute ORDER BY trip_start_minute)
  FROM UNNEST(dist_start_minute) AS trip_start_minute
)  ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_end_minute = " || trip_end_minute || ") AS trip_end_minute_" || trip_end_minute ORDER BY trip_end_minute)
  FROM UNNEST(dist_end_minute) AS trip_end_minute
) ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_start_hour = " || trip_start_hour || ") AS trip_start_hour_" || trip_start_hour ORDER BY trip_start_hour)
  FROM UNNEST(dist_start_hour) AS trip_start_hour
) ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_end_hour = " || trip_end_hour || ") AS trip_end_hour_" || trip_end_hour ORDER BY trip_end_hour)
  FROM UNNEST(dist_end_hour) AS trip_end_hour
) ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_start_day = " || trip_start_day || ") AS trip_start_day_" || trip_start_day ORDER BY trip_start_day)
  FROM UNNEST(dist_start_day) AS trip_start_day
) ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_end_day = " || trip_end_day || ") AS trip_end_day_" || trip_end_day ORDER BY trip_end_day)
  FROM UNNEST(dist_end_day) AS trip_end_day
) ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_start_month = " || trip_start_month || ") AS trip_start_month_" || trip_start_month ORDER BY trip_start_month)
  FROM UNNEST(dist_start_month) AS trip_start_month
) ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_end_month = " || trip_end_month || ") AS trip_end_month_" || trip_end_month ORDER BY trip_end_month)
  FROM UNNEST(dist_end_month) AS trip_end_month
) ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_start_year = " || trip_start_year || ") AS trip_start_year_" || trip_start_year ORDER BY trip_start_year)
  FROM UNNEST(dist_start_year) AS trip_start_year
) ||
 (
  SELECT ', ' || STRING_AGG("COUNTIF(trip_end_year = " || trip_end_year || ") AS trip_end_year_" || trip_end_year ORDER BY trip_end_year)
  FROM UNNEST(dist_end_year) AS trip_end_year
)|| '''
FROM `bt-int-ml-specialization.demo1.taxi_trips_preproc`
GROUP BY unique_key,
  trip_seconds_standardized,
  trip_miles_standardized,
  tips_standardized,
  extras_standardized,
  trip_total_standardized,
  label
''';  -- added `;`

CREATE OR REPLACE TABLE `bt-int-ml-specialization.demo1.taxi_trips_model_input` AS
SELECT * FROM result;  -- added line



CREATE OR REPLACE TABLE `bt-int-ml-specialization.demo1.taxi_trips_model_input_train` AS
SELECT * except (row_num)
FROM `bt-int-ml-specialization.demo1.taxi_trips_model_input`
WHERE row_num <= (SELECT COUNT(*) * 0.8 FROM `bt-int-ml-specialization.demo1.taxi_trips_model_input`);

CREATE OR REPLACE TABLE `bt-int-ml-specialization.demo1.taxi_trips_model_input_eval` AS
SELECT *  except (row_num)
FROM `bt-int-ml-specialization.demo1.taxi_trips_model_input`
WHERE row_num > (SELECT COUNT(*) * 0.8 FROM `bt-int-ml-specialization.demo1.taxi_trips_model_input`);

-- extract tables to cloud storage
-- bq extract --destination_format=PARQUET 'bt-int-ml-specialization:demo1.taxi_trips_model_input_train' gs://bt-int-ml-specialization-ml-demo1/data/taxi_trips/taxi_trips_model_input_train.parquet
-- bq extract --destination_format=PARQUET 'bt-int-ml-specialization:demo1.taxi_trips_model_input_eval' gs://bt-int-ml-specialization-ml-demo1/data/taxi_trips/taxi_trips_model_input_eval.parquet