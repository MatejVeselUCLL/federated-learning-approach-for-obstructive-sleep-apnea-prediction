CREATE TABLE apnea
(
  apnea_measurement_id serial NOT NULL,
  person_id smallint,
  measurement_datetime timestamp,
  person_source_value varchar(5),
  value_as_number_apnea double precision,
  value_as_number_spo2 double precision,
  value_as_number_hr double precision,
  CONSTRAINT apnea_pkey PRIMARY KEY (apnea_measurement_id)
);

---- VIEWS

DROP VIEW IF EXISTS apnea_groups CASCADE; DROP TABLE IF EXISTS h1 CASCADE; DROP TABLE IF EXISTS h2 CASCADE; DROP TABLE IF EXISTS h3 CASCADE;

CREATE OR REPLACE VIEW apnea_with_groups AS
SELECT *, substring(person_source_value FOR 1) "group"
FROM apnea;

CREATE OR REPLACE VIEW apnea_groups AS
SELECT DISTINCT person_id, person_source_value, substring(person_source_value FOR 1) "group"
FROM apnea
ORDER BY person_source_value;

CREATE OR REPLACE VIEW apnea_groups_no_validation AS
SELECT *
FROM apnea_groups
WHERE person_source_value NOT IN ('C15', 'C7', 'C13', 'C29', 'C11', 'C21', 'C32', 'D28', 'D33', 'D25', 'D5', 'D8', 'D13', 'ND5', 'ND1', 'ND7', 'ND2');

CREATE OR REPLACE VIEW count_patients_per_group AS
SELECT "group", count(1) count_patients 
FROM apnea_groups
GROUP BY "group"
ORDER BY 2 desc;

--SELECT *, (count_patients / 5) fifth_of_count_patients_floored
--FROM count_patients_per_group;


---- EXPLORATORY ANALSYS

-- Count of apnea measurements a person has.
--SELECT "group", person_source_value, person_id, sum(value_as_number_apnea)
--FROM apnea_with_groups
--WHERE value_as_number_apnea = 1.0
--GROUP BY "group", person_source_value, person_id
--ORDER BY 1, 4 ASC;


---- CREATING HOSPITAL DATABASES

-- Patients for hospital 1.
CREATE TABLE h1 AS (
WITH vars AS (SELECT 
 11 n_c, 
 10 n_d, 
 3 n_n)
-- Validation
(SELECT *
FROM apnea_groups
WHERE person_source_value IN ('C15','C7','C13','D28','D33','ND5','ND1'))
UNION
-- Group C
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'C' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_c FROM vars) ROWS ONLY)
UNION
-- Group D
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'D' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_d FROM vars) ROWS ONLY)
UNION
-- Group N
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'N' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_n FROM vars) ROWS ONLY)
ORDER BY person_source_value);

-- Patients for hospital 2.
CREATE TABLE h2 AS (
WITH vars AS (SELECT 
 10 n_c, 
 9 n_d, 
 2 n_n)
-- Validation
(SELECT *
FROM apnea_groups
WHERE person_source_value IN ('C29','C11','D25','D5','ND7'))
UNION
-- Group C
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'C' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_c FROM vars) ROWS ONLY)
UNION
-- Group D
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'D' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_d FROM vars) ROWS ONLY)
UNION
-- Group N
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'N' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_n FROM vars) ROWS ONLY)
ORDER BY person_source_value);

-- Patients for hospital 3.
CREATE TABLE h3 AS (
WITH vars AS (SELECT 
 10 n_c, 
 9 n_d, 
 2 n_n)
-- Validation
(SELECT *
FROM apnea_groups
WHERE person_source_value IN ('C21','C32','D8','D13','ND2'))
UNION
-- Group C
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'C' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_c FROM vars) ROWS ONLY)
UNION
-- Group D
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'D' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_d FROM vars) ROWS ONLY)
UNION
-- Group N
(SELECT *
FROM apnea_groups_no_validation
WHERE "group" = 'N' 
ORDER BY RANDOM()
FETCH FIRST (SELECT n_n FROM vars) ROWS ONLY)
ORDER BY person_source_value);

-- Verify
SELECT (SELECT count(*) from h1) + (SELECT count(*) from h2) + (SELECT count(*) from h3);

---- POPULATING HOSPITAL DATABASES




