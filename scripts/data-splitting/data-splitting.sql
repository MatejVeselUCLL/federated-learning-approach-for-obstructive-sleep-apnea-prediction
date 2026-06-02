-- README:
--  - Create a local Postgres database,
--  - execute the create table command,
--  - import data from the apnea csv (I use DBeaver's interface),
--  - (configure the script (with the help of data-splitting.xlsx). Configurable values/lines have comment '-- config')
--  - run this whole script,
--  - verify the created tables (h1, h2, h3). Verification commands are in the section commented as '-- Verify'
--  - export data from the created tables to csv (I use DBeaver's interface).

--CREATE TABLE apnea
--(
--  apnea_measurement_id serial NOT NULL,
--  person_id smallint,
--  measurement_datetime timestamp,
--  person_source_value varchar(5),
--  value_as_number_apnea double precision,
--  value_as_number_spo2 double precision,
--  value_as_number_hr double precision,
--  CONSTRAINT apnea_pkey PRIMARY KEY (apnea_measurement_id)
--);

---- VIEWS


DROP VIEW IF EXISTS apnea_with_groups CASCADE;
CREATE OR REPLACE VIEW apnea_with_groups AS
SELECT *, substring(person_source_value FOR 1) "group"
FROM apnea;

DROP VIEW IF EXISTS apnea_groups CASCADE;
CREATE OR REPLACE VIEW apnea_groups AS
SELECT DISTINCT person_id, person_source_value, substring(person_source_value FOR 1) "group"
FROM apnea
ORDER BY person_source_value;

DROP VIEW IF EXISTS apnea_groups_no_validation CASCADE;
CREATE OR REPLACE VIEW apnea_groups_no_validation AS
SELECT *
FROM apnea_groups
WHERE person_source_value NOT IN ('C15', 'C7', 'C13', 'C29', 'C11', 'C21', 'C32', 'D28', 'D33', 'D25', 'D5', 'D8', 'D13', 'ND5', 'ND1', 'ND7', 'ND2'); -- config

DROP VIEW IF EXISTS count_patients_per_group CASCADE;
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
DROP MATERIALIZED VIEW IF EXISTS h1_patients CASCADE;
CREATE MATERIALIZED VIEW h1_patients AS (
WITH vars AS (SELECT
 11 n_c, -- config
 10 n_d, -- config
 3 n_n) -- config
-- Validation
(SELECT *
FROM apnea_groups
WHERE person_source_value IN ('C15','C7','C13','D28','D33','ND5','ND1')) -- config
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
DROP materialized VIEW IF EXISTS h2_patients;
CREATE materialized VIEW h2_patients AS (
WITH vars AS (SELECT
 10 n_c, -- config
 9 n_d, -- config
 2 n_n), -- config
 other_hospitals_patients AS (SELECT person_source_value FROM h1_patients)
-- Validation
(SELECT *
FROM apnea_groups
WHERE person_source_value IN ('C29','C11','D25','D5','ND7')) -- config
UNION
-- Group C
(SELECT *
FROM apnea_groups_no_validation
WHERE
	person_source_value NOT IN (SELECT person_source_value FROM other_hospitals_patients) AND
	"group" = 'C'	
ORDER BY RANDOM()
FETCH FIRST (SELECT n_c FROM vars) ROWS ONLY)
UNION
-- Group D
(SELECT *
FROM apnea_groups_no_validation
WHERE 
	"group" = 'D' AND
	person_source_value NOT IN (SELECT person_source_value FROM other_hospitals_patients)
ORDER BY RANDOM()
FETCH FIRST (SELECT n_d FROM vars) ROWS ONLY)
UNION
-- Group N
(SELECT *
FROM apnea_groups_no_validation
WHERE 
	"group" = 'N' AND
	person_source_value NOT IN (SELECT person_source_value FROM other_hospitals_patients)
ORDER BY RANDOM()
FETCH FIRST (SELECT n_n FROM vars) ROWS ONLY)
ORDER BY person_source_value);

-- Patients for hospital 3.
DROP materialized VIEW IF EXISTS h3_patients;
CREATE materialized VIEW h3_patients AS (
WITH vars AS (SELECT 
 10 n_c, -- config
 9 n_d, -- config
 2 n_n), -- config
 other_hospitals_patients AS (
 	SELECT person_source_value FROM h1_patients
 	UNION
 	SELECT person_source_value FROM h2_patients
 )
-- Validation
(SELECT *
FROM apnea_groups
WHERE person_source_value IN ('C21','C32','D8','D13','ND2')) -- config
UNION
-- Group C
(SELECT *
FROM apnea_groups_no_validation
WHERE 
	"group" = 'C' AND
	person_source_value NOT IN (SELECT person_source_value FROM other_hospitals_patients)
ORDER BY RANDOM()
FETCH FIRST (SELECT n_c FROM vars) ROWS ONLY)
UNION
-- Group D
(SELECT *
FROM apnea_groups_no_validation
WHERE 
	"group" = 'D' AND
	person_source_value NOT IN (SELECT person_source_value FROM other_hospitals_patients)
ORDER BY RANDOM()
FETCH FIRST (SELECT n_d FROM vars) ROWS ONLY)
UNION
-- Group N
(SELECT *
FROM apnea_groups_no_validation
WHERE 
	"group" = 'N' AND
	person_source_value NOT IN (SELECT person_source_value FROM other_hospitals_patients)
ORDER BY RANDOM()
FETCH FIRST (SELECT n_n FROM vars) ROWS ONLY)
ORDER BY person_source_value);

-- Verify
--SELECT (SELECT count(*) from h1_patients) + (SELECT count(*) from h2_patients) + (SELECT count(*) from h3_patients);

---- POPULATING HOSPITAL DATABASES

-- Data for hospital 1.
DROP TABLE IF EXISTS h1 CASCADE;
CREATE TABLE h1 AS (
SELECT * FROM apnea
WHERE person_source_value IN (SELECT person_source_value FROM h1_patients));

-- Data for hospital 2.
DROP TABLE IF EXISTS h2 CASCADE;
CREATE TABLE h2 AS (
SELECT * FROM apnea
WHERE person_source_value IN (SELECT person_source_value FROM h2_patients));

-- Data for hospital 3.
DROP TABLE IF EXISTS h3 CASCADE;
CREATE TABLE h3 AS (
SELECT * FROM apnea
WHERE person_source_value IN (SELECT person_source_value FROM h3_patients));

DROP VIEW IF EXISTS apnea_from_hospitals CASCADE;
CREATE OR REPLACE VIEW apnea_from_hospitals AS (
	(SELECT *
	FROM h1)
	UNION ALL 
	(SELECT *
	FROM h2)
	UNION ALL
	(SELECT *
	FROM h3)
);

DROP VIEW IF EXISTS apnea_from_hospitals_ids CASCADE;
CREATE OR REPLACE VIEW apnea_from_hospitals_ids AS (
	(SELECT DISTINCT(person_source_value)
	FROM h1)
	UNION ALL 
	(SELECT DISTINCT(person_source_value)
	FROM h2)
	UNION ALL
	(SELECT DISTINCT(person_source_value)
	FROM h3)
);

-- Verify
SELECT
	(SELECT count(*) FROM apnea) count_measurements,
	(SELECT count(*) from h1) + (SELECT count(*) from h2) + (SELECT count(*) from h3) count_hospital_measurements,
	(SELECT count(DISTINCT person_source_value) from apnea) count_distinct_patients,
	((SELECT count(DISTINCT person_source_value) from h1) + (SELECT count(DISTINCT person_source_value) from h2) + (SELECT count(DISTINCT person_source_value) from h3)) count_distinct_patients_hospitals
	;

SELECT count(DISTINCT person_source_value)
FROM apnea_from_hospitals;




