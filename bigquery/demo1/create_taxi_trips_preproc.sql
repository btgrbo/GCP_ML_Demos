CREATE OR REPLACE TABLE `bt-int-ml-specialization.demo1.taxi_trips_preproc` AS
select
  unique_key,
  ifnull(replace(payment_type,' ', '_'), '99') as payment_type,
  format_date('%A', trip_start_timestamp) AS day_name,
  case when trip_start_timestamp is null then 99 else extract(minute from trip_start_timestamp) end as trip_start_minute,
  case when trip_end_timestamp is null then 99 else extract(minute from trip_end_timestamp) end as trip_end_minute,
  case when trip_start_timestamp is null then 99 else extract(hour from trip_start_timestamp) end as trip_start_hour,
  case when trip_end_timestamp is null then 99 else extract(hour from trip_end_timestamp) end as trip_end_hour,
  case when trip_start_timestamp is null then 99 else extract(day from trip_start_timestamp) end as trip_start_day,
  case when trip_end_timestamp is null then 99 else extract(day from trip_end_timestamp) end as trip_end_day,
  case when trip_start_timestamp is null then 99 else extract(month from trip_start_timestamp) end as trip_start_month,
  case when trip_end_timestamp is null then 99 else extract(month from trip_end_timestamp) end as trip_end_month,
  case when trip_start_timestamp is null then 99 else extract(year from trip_start_timestamp) end as trip_start_year,
  case when trip_end_timestamp is null then 99 else extract(year from trip_end_timestamp) end as trip_end_year,
  ifnull(ml.standard_scaler(trip_seconds) over(), 0) as trip_seconds_standardized,
  ifnull(ml.standard_scaler(trip_miles) over(), 0) as trip_miles_standardized,
  ifnull(ml.standard_scaler(tips) over(), 0) as tips_standardized,
  ifnull(ml.standard_scaler(extras) over(), 0) as extras_standardized,
  ifnull(ml.standard_scaler(trip_total) over(), 0) as trip_total_standardized,
  ifnull(fare, AVG(fare) over ()) as label
FROM `bt-int-ml-specialization.demo1.taxi_trips`
LIMIT 5000;