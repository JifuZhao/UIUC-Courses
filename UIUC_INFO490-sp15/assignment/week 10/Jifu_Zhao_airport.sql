
-- To run the whole airport.sql, you may need a quite while
-- Please be patient

-- This is going to drop the table that exists
-- If there is no table exist, it will return error, but it doesn't matter

DROP TABLE IF EXISTS flights;
DROP TABLE IF EXISTS iata;
DROP TABLE IF EXISTS myTable;
DROP TABLE IF EXISTS template;

CREATE TABLE flights (
    year INT,
    month INT,
    dayOfMonth INT,
    dayOfWeek INT,
    actualDepartureTime INT,
    scheduledDepartureTime INT,
    arrivalArrivalTime INT,
    scheduledArrivalTime INT,
    uniqueCarrierCode TEXT,
    flightNumber INT,
    tailNumber TEXT,
    actualElapsedTime INT,
    scheduledElapsedTime INT,
    airTime INT,
    arrivalDelay INT,
    departureDelay INT,
    originCode TEXT,
    destinationCode TEXT,
    distance INT,
    taxiIn INT,
    taxiOut INT,
    cancelled INT,
    cancellationCode TEXT,
    diverted TEXT,
    carrierDelay INT,
    weatherDelay INT,
    nasDelay INT,
    securityDelay INT,
    lateAircraftDelay INT
);

CREATE TABLE iata (
    airportID INT,
    name TEXT,
    city TEXT,
    country TEXT,
    iata TEXT,
    icao TEXT,
    latitude REAL,
    longitude REAL,
    altitude REAL,
    timeZone INT,
    dst TEXT,
    tzDatabaseTimeZone TEXT
);

-- Note: this is going to import data
-- you may need to change the path of 2001.csv and iata.csv

.mode csv
.import /notebooks/i2ds/data/2001.csv flights
.import /notebooks/i2ds/spring2015/week10/iata.csv iata

DELETE FROM flights WHERE Year = 'Year';

SELECT COUNT(*) AS COUNT FROM flights;

SELECT COUNT(*) AS COUNT FROM iata;

CREATE TABLE template
    AS SELECT month AS 'month', dayOfMonth AS 'dayOfMonth', uniqueCarrierCode AS 'uniqueCarrierCode',
                flightNumber AS 'flightNumber', scheduledDepartureTime AS 'scheduledDepartureTime',
                diverted AS 'diverted', destinationCode AS 'destinationCode', iata.city AS 'originCity' 
        FROM flights LEFT OUTER JOIN iata ON flights.originCode = iata.iata;
        
CREATE TABLE myTable
    AS SELECT month AS 'month', dayOfMonth AS 'dayOfMonth', uniqueCarrierCode AS 'uniqueCarrierCode',
                flightNumber AS 'flightNumber', scheduledDepartureTime AS 'scheduledDepartureTime',
                diverted AS 'diverted', originCity AS 'originCity', iata.city AS 'destinationCity'
        FROM template LEFT OUTER JOIN iata ON template.destinationCode = iata.iata;

SELECT COUNT(*) AS COUNT FROM myTable;

INSERT INTO myTable (month, dayOfMonth, uniqueCarrierCode, flightNumber,
                     scheduledDepartureTime, diverted, originCity, destinationCity)
VALUES(9, 9, 'INFO', 490, 0800, 1, 'Champaign', 'Chicago');

SELECT COUNT(*) AS COUNT FROM myTable;

SELECT AVG(f.departureDelay) AS Average FROM flights AS f WHERE f.departureDelay NOT LIKE 'NA';
SELECT MAX(f.departureDelay) AS Maximum FROM flights AS f WHERE f.departureDelay NOT LIKE 'NA';
SELECT Min(f.departureDelay) AS Minimum FROM flights AS f WHERE f.departureDelay NOT LIKE 'NA';

SELECT f.month, f.dayOfMonth, f.uniqueCarrierCode, f.flightNumber FROM myTable AS f
    WHERE f.scheduledDepartureTime > 0745 AND f.scheduledDepartureTime < 0815 
        AND f.originCity = 'Newark' AND f.destinationCity = 'San Francisco' AND f.diverted = '1';