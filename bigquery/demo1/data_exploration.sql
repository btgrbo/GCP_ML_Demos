#### Data Quality Checks ####
# First quick check of 100 entries
SELECT * FROM `bt-int-ml-specialization.demo1.taxi_trips` LIMIT 100;

# Check for duplicate rows?
SELECT
  COUNT(DISTINCT unique_key) as distinct_entries,
  count(*) as n_rows
FROM `bt-int-ml-specialization.demo1.taxi_trips`;

/*
  "distinct_entries": "211655493",
  "n_rows": "211655493"
-> no duplicate rows
*/

# Check for 0 or negative entries
SELECT
  COUNTIF(fare <= 0) / count(*) * 100  as perc_zero_neg_fare,
  COUNTIF(trip_seconds <= 0) / count(*) * 100 as perc_zero_neg_trip_seconds,
  COUNTIF(trip_miles <= 0) / count(*) * 100 as perc_zero_neg_trip_miles
FROM `bt-int-ml-specialization.demo1.taxi_trips`;

# Check for trip_end_timestamp <= trip_start_timestamp
SELECT
  COUNTIF(trip_end_timestamp <= trip_start_timestamp) / count(*) * 100  as perc_zero_neg_fare
FROM `bt-int-ml-specialization.demo1.taxi_trips`;


SELECT
  'fare' as field,
  COUNT(fare) AS count,
  AVG(fare) AS mean,
  STDDEV(fare) AS stddev,
  MIN(fare) AS min,
  MAX(fare) AS max
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`
UNION ALL
SELECT
  'trip_seconds' as field,
  COUNT(trip_seconds) AS count,
  AVG(trip_seconds) AS mean,
  STDDEV(trip_seconds) AS stddev,
  MIN(trip_seconds) AS min,
  MAX(trip_seconds) AS max
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`
UNION ALL
SELECT
  'trip_miles' as field,
  COUNT(trip_miles) AS count,
  AVG(trip_miles) AS mean,
  STDDEV(trip_miles) AS stddev,
  MIN(trip_miles) AS min,
  MAX(trip_miles) AS max
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`
UNION ALL
SELECT
  'pickup_latitude' as field,
  COUNT(pickup_latitude) AS count,
  AVG(pickup_latitude) AS mean,
  STDDEV(pickup_latitude) AS stddev,
  MIN(pickup_latitude) AS min,
  MAX(pickup_latitude) AS max
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`
UNION ALL
SELECT
  'pickup_longitude' as field,
  COUNT(pickup_longitude) AS count,
  AVG(pickup_longitude) AS mean,
  STDDEV(pickup_longitude) AS stddev,
  MIN(pickup_longitude) AS min,
  MAX(pickup_longitude) AS max
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`
UNION ALL
SELECT
  'dropoff_latitude' as field,
  COUNT(dropoff_latitude) AS count,
  AVG(dropoff_latitude) AS mean,
  STDDEV(dropoff_latitude) AS stddev,
  MIN(dropoff_latitude) AS min,
  MAX(dropoff_latitude) AS max
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`
UNION ALL
SELECT
  'dropoff_longitude' as field,
  COUNT(dropoff_longitude) AS count,
  AVG(dropoff_longitude) AS mean,
  STDDEV(dropoff_longitude) AS stddev,
  MIN(dropoff_longitude) AS min,
  MAX(dropoff_longitude) AS max
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`
  order by field;


SELECT
  'trip_start_timestamp' as field,
  COUNT(trip_start_timestamp) AS count,
  date(MIN(trip_start_timestamp)) AS first_occurrence,
  date(MAX(trip_start_timestamp)) AS last_occurrence,
  TIMESTAMP_DIFF(MAX(trip_start_timestamp), MIN(trip_start_timestamp), DAY) AS range_days
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`;


SELECT
  MIN(trip_start_timestamp),
  MAX(trip_start_timestamp)
from `bt-int-ml-specialization.demo1.taxi_trips`;


# Which columns does the table have?
SELECT column_name
FROM `bt-int-ml-specialization.demo1.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'taxi_trips';

/*
unique_key: not relevant for analysis
taxi_id: not relevant for analysis
trip_start_timestamp: relevant -> minute, hour, day, month, day of week
trip_end_timestamp: relevant -> minute, hour, day, month, day of week
trip_seconds: not relevant for analysis, because not known before the trip starts
trip_miles: not relevant for analysis, because not known before the trip starts
pickup_census_tract: relevant
dropoff_census_tract: relevant
pickup_community_area: relevant
dropoff_community_area: relevant
fare: label
tips: not relevant
tolls: not relevant
extras: not relevant
trip_total: not relevant
payment_type: not relevant
company: not relevant
pickup_latitude: relevant
pickup_longitude: relevant
pickup_location: not relevant included in pickup_longitude/latitude
dropoff_latitude: relevant
dropoff_longitude: relevant
dropoff_location: not relevant included in pickup_longitude/latitude
*/


# How many taxi_ids compared to taxi trips?
SELECT
  COUNT(DISTINCT taxi_id) as distinct_taxis,
  COUNT(DISTINCT unique_key) as distinct_trips
FROM `bt-int-ml-specialization.demo1.taxi_trips`;

/*
[{
  "distinct_taxis": "9805",
  "distinct_trips": "211655493"
}]
*/


# Are there any columns with too many missing values?
SELECT
  (count(*) - count(trip_start_timestamp)) / count(*) * 100 as perc_missing_trip_start_timestamp,
  #(count(*) - count(trip_end_timestamp)) / count(*) * 100 as perc_missing_trip_end_timestamp,
  #(count(*) - count(pickup_census_tract)) / count(*) * 100 as perc_missing_pickup_census_tract,
  #(count(*) - count(dropoff_census_tract)) / count(*) * 100 as perc_missing_dropoff_census_tract,
  #(count(*) - count(pickup_community_area)) / count(*) * 100 as perc_missing_pickup_community_area,
  #(count(*) - count(dropoff_community_area)) / count(*) * 100 as perc_missing_dropoff_community_area,
  (count(*) - count(fare)) / count(*) * 100 as perc_missing_fare,
  (count(*) - count(trip_seconds)) / count(*) * 100 as perc_missing_trip_seconds,
  (count(*) - count(trip_miles)) / count(*) * 100 as perc_missing_trip_miles,
  (count(*) - count(pickup_latitude)) / count(*) * 100 as perc_missing_pickup_latitude,
  (count(*) - count(pickup_longitude)) / count(*) * 100 as perc_missing_pickup_longitude,
  (count(*) - count(dropoff_latitude)) / count(*) * 100 as perc_missing_dropoff_latitude,
  (count(*) - count(dropoff_longitude)) / count(*) * 100 as perc_missing_dropoff_longitude
FROM `bt-int-ml-specialization.demo1.taxi_trips`;

/*
[{
  "perc_missing_trip_start_timestamp": "0.0",
  "perc_missing_trip_end_timestamp": "0.0087259724461769585",
  "perc_missing_pickup_census_tract": "36.799914283349125",
  "perc_missing_dropoff_census_tract": "37.254324412927005",
  "perc_missing_pickup_community_area": "11.422305491499813",
  "perc_missing_dropoff_community_area": "13.516277604947394",
  "perc_missing_fare": "0.0099491866247005453",
  "perc_missing_pickup_latitude": "11.409348114579762",
  "perc_missing_pickup_longitude": "11.409348114579762",
  "perc_missing_dropoff_latitude": "13.258188862596635",
  "perc_missing_dropoff_longitude": "13.258188862596635"
}]
*/

SELECT
  COUNTIF(tolls IS NULL) AS missing_count,
  COUNTIF(tolls = 0) AS zero_count,
  COUNT(*) AS total_count,
  (COUNTIF(tolls IS NULL) + COUNTIF(tolls = 0)) / count(*) * 100 as perc_zero_missing
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`;


SELECT
  COUNTIF(extras IS NULL) AS missing_count,
  COUNTIF(extras = 0) AS zero_count,
  COUNT(*) AS total_count,
  (COUNTIF(extras IS NULL) + COUNTIF(extras = 0)) / count(*) * 100 as perc_zero_missing
FROM
  `bt-int-ml-specialization.demo1.taxi_trips`;
