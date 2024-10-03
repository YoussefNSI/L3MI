-- SQLBook: Code
CREATE SEQUENCE IF NOT EXISTS idprod;
CREATE TABLE IF NOT EXISTS PRODUIT (
NumProd INTEGER PRIMARY KEY,
Designation VARCHAR(50),
Prix FLOAT,
quantite INTEGER);


CREATE OR REPLACE FUNCTION creer_produit2()
RETURNS VOID AS $$
DECLARE
    cur_prod CURSOR FOR SELECT * FROM PRODUIT;
    rec_prod RECORD;
    nouveauPrix NUMERIC;
    BEGIN
        CREATE TABLE IF NOT EXISTS PRODUIT2(
        NumProd INTEGER PRIMARY KEY,
        Designation VARCHAR(255),
        Prix NUMERIC(2),
        Quantite INTEGER
        );

        TRUNCATE TABLE PRODUIT2;

        IF NOT EXISTS (SELECT * FROM PRODUIT) THEN
            INSERT INTO PRODUIT2 VALUES(0, 'Pas de produit', NULL, NULL);
        ELSE
            OPEN cur_prod;
            LOOP
                FETCH cur_prod INTO rec_prod;

                IF rec_prod.Prix IS NULL THEN
                    nouveauPrix := 0;
                ELSE
                    nouveauPrix := rec_prod.Prix;
                END IF;

                IF rec_prod.quantite > 50 THEN
                    nouveauPrix := nouveauPrix * 0.30;
                ELSIF rec_prod.quantite < 10 AND nouveauPrix > 400 THEN
                    nouveauPrix := nouveauPrix * 1.60;
                END IF;

                EXIT WHEN NOT FOUND;

                INSERT INTO PRODUIT2(NumProd, Designation, Prix, Quantite)
                VALUES(rec_prod.NumProd, UPPER(rec_prod.Designation), nouveauPrix, rec_prod.Quantite);
            END LOOP;
            CLOSE cur_prod;
        END IF;
END;

$$ LANGUAGE plpgsql;

CREATE SEQUENCE IF NOT EXISTS avid;
CREATE TABLE IF NOT EXISTS AVION(
AvNum INTEGER PRIMARY KEY,
type VARCHAR(50)
);

CREATE SEQUENCE IF NOT EXISTS plid;
CREATE TABLE IF NOT EXISTS PILOTE(
PlNum INTEGER PRIMARY KEY,
PlNom VARCHAR(50),
PlPrenom VARCHAR(50)
);

CREATE SEQUENCE IF NOT EXISTS volid;
CREATE TABLE IF NOT EXISTS VOL(
VolNum INTEGER PRIMARY KEY,
PlNum INTEGER REFERENCES PILOTE(PlNum),
AvNum INTEGER REFERENCES AVION(AvNum),
HeureDep TIME,
HeureArr TIME
);

CREATE OR REPLACE FUNCTION MajVol()
RETURNS VOID AS $$
DECLARE
    curVol14 CURSOR FOR SELECT * FROM VOL WHERE VolNum == 1 OR VolNum == 4;
    recVol RECORD;
    tpsVol INTEGER;
    nouvelHeureArr TIME;
    BEGIN
        IF EXISTS(SELECT * FROM VOL WHERE VolNum == 1 OR VolNum == 4) THEN
            OPEN curVol14;
            LOOP
                FETCH curVol14 INTO recVol;
                EXIT WHEN NOT FOUND;
                tpsVol = recVol.HeureArr.TIME_TO_SEC() - recVol.HeureDep.TIME_TO_SEC();
                tpsVol * 0.90;
                nouvelHeureArr = SEC_TO_TIME(recVol.HeureDep.TIME_TO_SEC() + tpsVol);
                UPDATE VOL SET HeureArr = nouvelHeureArr WHERE VolNum = recVol.VolNum;
            END LOOP;
        END IF;
END;

$$ LANGUAGE plpgsql;


