-- Business/analytics questions for the project
-- SQLite compatible.

-- 1. Average wage by position
SELECT
    position,
    ROUND(AVG(wage), 2) AS avg_wage,
    COUNT(*) AS player_count
FROM players
GROUP BY position
ORDER BY avg_wage DESC;

-- 2. Average wage by league
SELECT
    league,
    ROUND(AVG(wage), 2) AS avg_wage,
    COUNT(*) AS player_count
FROM players
GROUP BY league
ORDER BY avg_wage DESC;

-- 3. Top 10 highest-paid players by club record
SELECT
    club,
    position,
    age,
    wage
FROM players
ORDER BY wage DESC
LIMIT 10;

-- 4. Top 3 highest-paid players within each league
WITH ranked AS (
    SELECT
        player_id,
        club,
        league,
        position,
        wage,
        ROW_NUMBER() OVER (
            PARTITION BY league
            ORDER BY wage DESC
        ) AS rn
    FROM players
)
SELECT *
FROM ranked
WHERE rn <= 3
ORDER BY league, wage DESC;

-- 5. International exposure bands
SELECT
    CASE
        WHEN international_exposure < 0.05 THEN 'Low'
        WHEN international_exposure < 0.15 THEN 'Medium'
        ELSE 'High'
    END AS exposure_band,
    ROUND(AVG(wage), 2) AS avg_wage,
    COUNT(*) AS player_count
FROM players
GROUP BY exposure_band
ORDER BY avg_wage DESC;

-- 6. Wage by age band
SELECT
    CASE
        WHEN age < 21 THEN 'U21'
        WHEN age BETWEEN 21 AND 24 THEN '21-24'
        WHEN age BETWEEN 25 AND 28 THEN '25-28'
        WHEN age BETWEEN 29 AND 32 THEN '29-32'
        ELSE '33+'
    END AS age_band,
    ROUND(AVG(wage), 2) AS avg_wage,
    COUNT(*) AS player_count
FROM players
GROUP BY age_band
ORDER BY avg_wage DESC;

-- 7. High-app players vs wage
SELECT
    CASE
        WHEN apps >= 250 THEN '250+ Apps'
        WHEN apps >= 100 THEN '100-249 Apps'
        ELSE '<100 Apps'
    END AS appearance_band,
    ROUND(AVG(wage), 2) AS avg_wage,
    COUNT(*) AS player_count
FROM players
GROUP BY appearance_band
ORDER BY avg_wage DESC;
