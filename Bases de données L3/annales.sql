
CREATE TYPE pilote AS (
    num_pilote VARCHAR(5),
    nom VARCHAR(30),
    prenom VARCHAR(30)
);

CREATE TABLE tpilote OF pilote (
    PRIMARY KEY (num_pilote)
);

CREATE TYPE passager AS (
    num_pass VARCHAR(5),
    nom VARCHAR(30),
    prenom VARCHAR(30)
);

CREATE TABLE tpassager OF passager (
    PRIMARY KEY (num_pass)
);

CREATE TYPE avion AS (
    num_avion VARCHAR(5),
    type VARCHAR(30),
    capacite INTEGER
);

CREATE TABLE tavion OF avion (
    PRIMARY KEY (num_avion)
);

CREATE TYPE trajet AS (
    villeDepart VARCHAR(30),
    villeArrivee VARCHAR(30)
);

CREATE TABLE ttrajet OF trajet;

CREATE TYPE vol AS (
    numVol VARCHAR(5),
    pilote tpilote,
    trajet ttrajet,
    avion tavion,
    dateVol DATE
);

CREATE TABLE tvol OF vol;

CREATE TYPE reservation AS (
    num_res VARCHAR(5),
    passager tpassager,
    vol tvol,
    num_Place VARCHAR(5)
);

CREATE TABLE treservation OF reservation;

-- 1.2

INSERT INTO tpilote VALUES ('p1', 'Dupond', 'Phillipe');
INSERT INTO tpilote VALUES ('p2', 'Hamon', 'Elea');

INSERT INTO tpassager VALUES ('PS001', 'Keyes', 'Daniel');
INSERT INTO tpassager VALUES ('PS002', 'Levey', 'Anna');

INSERT INTO tavion VALUES ('AV01', 'Boeing 777', 2);
INSERT INTO tavion VALUES ('AV02', 'Airbus A330', 3);

INSERT INTO ttrajet VALUES ('Paris', 'New York');
INSERT INTO ttrajet VALUES ('Nice', 'Tokyo');

INSERT INTO tvol VALUES ('AF232', ROW('p2', 'Hamon', 'Elea'), ROW('Nice', 'Tokyo'), ROW('AV02', 'Airbus A330', 3), '2020-01-02');

INSERT INTO treservation(num_res, passager, num_Place) VALUES ('RB27', ROW('PS002', 'Levey', 'Anna'), 'A21');
INSERT INTO treservation(num_res, passager, num_Place) VALUES ('RB28', ROW('PS001', 'Keyes', 'Daniel'), 'A22');

-- 1.3

INSERT INTO tvol VALUES ('AF231', ROW('p1', 'Dupond', 'Phillipe'), ROW('Paris', 'New York'), ROW('AV01', 'Boeing 777', 2), '2021-03-31');

-- 1.4

INSERT INTO treservation(num_res, passager, num_Place) VALUES ('RB27', ROW('PS002', 'Levey', 'Anna'), 'A21');

-- 1.5

UPDATE treservation SET vol = (select a from tvol a where numVol='AF231') WHERE num_res = 'RB27';

-- 1.6

DELETE FROM tvol WHERE (tvol.pilote).num_pilote = 'p1';

-- 1.7

CREATE OR REPLACE FUNCTION f_stats_pilote(numPilote VARCHAR(5))
RETURNS VOID AS $$
DECLARE
    cur CURSOR FOR SELECT * FROM tvol WHERE (tvol.pilote).num_pilote = numPilote;
    rec RECORD;
    nb_vols INTEGER := 0;
    dateLastTrajet DATE := NULL;
    villeLastTrajet VARCHAR(30) := NULL;
BEGIN
    OPEN cur;
    LOOP
        FETCH cur INTO rec;
        EXIT WHEN NOT FOUND;
        nb_vols := nb_vols + 1;
        dateLastTrajet := rec.dateVol;
        villeLastTrajet := (rec.trajet).villeArrivee;
    END LOOP;
    CLOSE cur;
    
    IF nb_vols = 0 THEN
        RAISE NOTICE 'Le pilote % n''a effectué aucun vol.', numPilote;
    ELSE
        RAISE NOTICE 'Le pilote % a effectué % vols, son dernier vol était le % à destination de %', 
            numPilote, nb_vols, dateLastTrajet, villeLastTrajet;
    END IF;
END;
$$ LANGUAGE plpgsql;


SELECT f_stats_pilote('p2');

-- 1.8

CREATE OR REPLACE FUNCTION f_pilote_ville(ville VARCHAR(30))
RETURNS VOID AS $$
DECLARE
    cur CURSOR FOR SELECT * FROM tvol WHERE (tvol.trajet).villeArrivee = ville;
    rec RECORD;
    pilotes pilote[];
    i INTEGER := 0;
    nbTrajets INTEGER := 0;
    nbTrajets_aux INTEGER := 0;
    pilotePlusActif pilote;
BEGIN
    OPEN cur;
    LOOP
        FETCH cur INTO rec;
        EXIT WHEN NOT FOUND;
        nbTrajets_aux := nbTrajets_aux + 1;
        IF i = 0 THEN
            i := i + 1;
            pilotes[i] := rec.pilote;
        ELSE
            FOR j IN 1..i LOOP
                IF pilotes[j].num_pilote = rec.pilote.num_pilote THEN
                    nbTrajets := nbTrajets + 1;
                ELSE
                    i := i + 1;
                    pilotes[i] := rec.pilote;
                END IF;
            END LOOP;
        END IF;
    END LOOP;
    CLOSE cur;
    
    IF nbTrajets_aux = 0 THEN
        RAISE NOTICE 'Aucun vol n''a atterri à %.', ville;
    ELSE
        FOR j IN 1..i LOOP
            IF nbTrajets < nbTrajets_aux THEN
                nbTrajets := nbTrajets_aux;
                pilotePlusActif := pilotes[j];
            END IF;
        END LOOP;
        RAISE NOTICE 'Le pilote le plus actif à % est % avec % vols.', ville, pilotePlusActif.nom, nbTrajets;
    END IF;
END;

$$ LANGUAGE plpgsql;

SELECT f_pilote_ville('Tokyo');

-- 1.9

CREATE OR REPLACE FUNCTION check_vol_retour()
RETURNS TRIGGER AS $$
BEGIN
    IF (NEW.trajet).villeArrivee != (NEW.trajet).villeDepart THEN
        RAISE EXCEPTION 'Le vol retour n''est pas présent dans la table';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER volRetour_present
BEFORE INSERT OR UPDATE ON tvol
FOR EACH ROW
EXECUTE FUNCTION check_vol_retour();

-- 1.10

CREATE OR REPLACE FUNCTION check_capacite()
RETURNS TRIGGER AS $$
DECLARE
    nbReservation INTEGER;
    capacite INTEGER;
BEGIN
    SELECT COUNT(*) INTO nbReservation FROM treservation WHERE (treservation.vol).numVol = (NEW.vol).numVol;
    SELECT (NEW.vol).capacite INTO capacite;
    IF nbReservation > capacite THEN
        RAISE EXCEPTION 'La capacité de l''avion est dépassée';
    END IF;
    RETURN NEW;
END;

$$ LANGUAGE plpgsql;

CREATE TRIGGER capacite_depasse
BEFORE UPDATE ON tvol
FOR EACH ROW
EXECUTE FUNCTION check_capacite();


