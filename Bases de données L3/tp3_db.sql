-- SQLBook: Code
CREATE TABLE IF NOT EXISTS Segment(
indIP INTEGER PRIMARY KEY,
nomSegment VARCHAR(50),
nbPoste INTEGER
);

CREATE TABLE IF NOT EXISTS Salle(
nsalle INTEGER PRIMARY KEY,
nomSalle VARCHAR(50),
nbPoste INTEGER,
indIP INTEGER REFERENCES Segment(indIP)
);

CREATE TABLE Poste(
nposte INTEGER PRIMARY KEY,
nomPoste VARCHAR(50),
typePoste VARCHAR(50),
nsalle INTEGER REFERENCES Salle(nsalle),
nblog INTEGER
);

CREATE TABLE IF NOT EXISTS Logiciel(
nlog INTEGER PRIMARY KEY,
nomLog VARCHAR(50),
dateAch DATE,
version VARCHAR(20),
typeLog VARCHAR(30),
prix NUMERIC,
nbInstal INTEGER
);

CREATE TABLE IF NOT EXISTS Installer(
nposte INTEGER REFERENCES Poste(nposte),
nLog INTEGER REFERENCES Logiciel(nlog),
numIns INTEGER,
dateIns DATE,
delai NUMERIC,
PRIMARY KEY (nposte, nLog)
);

/*
INSERT INTO Segment VALUES(130, 'Ethernet', 20);
INSERT INTO Salle VALUES(1, 'Ada Lovelace', 20, 130);
INSERT INTO Poste VALUES(1, 'Poste 1', 'Windows', 1, 1);
INSERT INTO Logiciel VALUES(1, 'Logiciel1', '2024-06-09', 'Type1', 99.99, 17500);
INSERT INTO Installer(nposte, nLog, numIns, dateIns) VALUES(1, 1, 1, '2024-09-16');
*/
CREATE OR REPLACE FUNCTION CalculTemps()
RETURNS VOID AS $$
DECLARE
    cur_ins CURSOR FOR SELECT * FROM INSTALLER;
    del INTEGER;
    ins_date DATE;
    rec_ins RECORD;
    BEGIN
        OPEN cur_ins;
        LOOP
            FETCH cur_ins INTO rec_ins;
            EXIT WHEN NOT FOUND;

            SELECT dateAch INTO ins_date FROM Logiciel WHERE nlog = rec_ins.nLog;
            IF EXISTS (SELECT dateAch INTO ins_date FROM Logiciel WHERE nlog = rec_ins.nLog) THEN
                del = rec_ins.nLog - ins_date;
            ELSE
                RAISE NOTICE 'date d achat inconnue';
            END IF;

            IF del >= 0 THEN
                UPDATE Installer
                SET delai = del
                WHERE nposte = rec_ins.nposte
                AND nLog = rec_ins.nLog;
                RAISE NOTICE 'OK delai de (%, %) modifié', rec_ins.nposte, rec_ins.nLog;
            ELSE
                RAISE NOTICE 'La date dinstallation est antérieure à la date d achat';
            END IF;
        END LOOP;
    CLOSE cur_ins;
END;

--- La fonction "installLogSeg" ne fonctionne pas encore.

CREATE OR REPLACE FUNCTION installLogSeg(
ipt VARCHAR,
logName VARCHAR,
typeLog VARCHAR,
logDate DATE,
logVersion VARCHAR,
logPlatform VARCHAR,
logPrice NUMERIC
)
RETURNS VOID AS $$
DECLARE
    v_nposte INTEGER;
    v_nsalle INTEGER;
    v_typePoste VARCHAR(30);
    v_numIns INTEGER;
    v_nbInstal INTEGER;
    CURSOR c_postes IS
    SELECT p.nposte, p.nsalle, p.typePoste
    FROM Poste p
    JOIN Salle s ON p.nsalle = s.nsalle
    JOIN Segment seg ON s.indIP = seg.indIP
    WHERE seg.indIP = ipt AND p.typePoste = typeLog;
    
BEGIN
    BEGIN
        INSERT INTO Logiciel (nlog, nomLog, dateAch, version, typeLog, prix, nbInstal)
        VALUES (DEFAULT, logName, logDate, logVersion, logPlatform, logPrice, 0)
        RETURNING nlog INTO v_numIns;
        RAISE NOTICE 'Logiciel % ajouté dans la table Logiciel', logName;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Erreur lors de l''insertion du logiciel % dans la table Logiciel', logName;
        RETURN;
    END;


OPEN c_postes;

LOOP
    FETCH c_postes INTO v_nposte, v_nsalle, v_typePoste;
    EXIT WHEN NOT FOUND;

    BEGIN

        INSERT INTO Installer (nposte, nlog, numIns, dateIns, delai)
        VALUES (v_nposte, v_numIns, v_numIns, CURRENT_DATE, NULL);

        UPDATE Logiciel
        SET nbInstal = nbInstal + 1
        WHERE nlog = v_numIns;

        RAISE NOTICE 'Installation sur Poste % dans la Salle %', v_nposte, v_nsalle;

    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'Erreur lors de l''installation sur le poste % dans la salle %', v_nposte, v_nsalle;
    END;
END LOOP;

CLOSE c_postes;

END;

$$ LANGUAGE plpgsql;


