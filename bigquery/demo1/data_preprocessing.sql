
CREATE OR REPLACE TABLE `bt-int-ml-specialization.demo1.taxi_trips_clean` AS
SELECT
  unique_key,
  trip_start_timestamp,
  fare,
  trip_seconds,
  trip_miles,
  pickup_latitude,
  pickup_longitude,
  dropoff_latitude,
  dropoff_longitude
FROM `bt-int-ml-specialization.demo1.taxi_trips`
WHERE
  trip_start_timestamp is not null
  and fare is not null
  and trip_seconds is not null
  and trip_miles is not null
  and pickup_latitude is not null
  and pickup_longitude is not null
  and dropoff_latitude is not null
  and dropoff_longitude is not null
  and fare > 0
  and trip_seconds > 0
  and trip_miles > 0
  and trip_end_timestamp > trip_start_timestamp;

CREATE OR REPLACE TABLE `bt-int-ml-specialization.demo1.taxi_trips_ex_outlier` AS
WITH rank_cte as(
  SELECT
    unique_key,
    RANK() OVER (ORDER BY RAND()) as rnd_rank,
    COUNT(*) OVER () as total_rows
    FROM `bt-int-ml-specialization.demo1.taxi_trips_clean`
)
  SELECT
    a.trip_start_timestamp,
    a.fare,
    a.trip_seconds,
    a.trip_miles,
    a.pickup_latitude,
    a.pickup_longitude,
    a.dropoff_latitude,
    a.dropoff_longitude,
    b.rnd_rank/b.total_rows as norm_rank
  FROM `bt-int-ml-specialization.demo1.taxi_trips_clean` a
  INNER JOIN rank_cte b
  ON a.unique_key = b.unique_key
  WHERE a.fare < 80
  AND a.trip_seconds < 6000
  AND a.trip_miles < 30
  LIMIT 5000000;

CREATE OR REPLACE TABLE bt-int-ml-specialization.demo1.taxi_trips_train AS
SELECT * EXCEPT(norm_rank)
FROM bt-int-ml-specialization.demo1.taxi_trips_ex_outlier
WHERE norm_rank <= 0.8;

CREATE OR REPLACE TABLE bt-int-ml-specialization.demo1.taxi_trips_eval AS
SELECT * EXCEPT(norm_rank)
FROM bt-int-ml-specialization.demo1.taxi_trips_ex_outlier
WHERE norm_rank >= 0.8;
