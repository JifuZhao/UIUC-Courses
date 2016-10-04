
DROP TABLE flights;

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


.mode csv
.import /notebooks/i2ds/data/2001.csv flights

DELETE FROM flights WHERE Year = 'Year';

SELECT COUNT(*) AS COUNT FROM flights;