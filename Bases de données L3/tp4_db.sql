--- Exercice 1

-- 1.1

CREATE TYPE tadmin AS (
    nadmin INTEGER,
    nom VARCHAR(30),
    age INTEGER
);

CREATE TABLE admin OF tadmin (
    PRIMARY KEY (nadmin)
);

CREATE TYPE tPC AS (
    numSerie INTEGER,
    adresseIP VARCHAR(15),
    nadmin INTEGER
);

CREATE TABLE PC OF tPC (
    PRIMARY KEY (numSerie),
    FOREIGN KEY (nadmin) REFERENCES admin
);

-- 1.2

INSERT INTO admin VALUES (1, 'admin1', 20);
INSERT INTO admin VALUES (2, 'admin2', 30);

INSERT INTO PC(numSerie, adresseIP) VALUES (1, '1900.0.0.1');
INSERT INTO PC(numSerie, adresseIP) VALUES (2, '1900.0.0.2');

-- 1.3

INSERT INTO PC(numSerie, adresseIP, nadmin) VALUES (3, '1900.0.0.3', 1);

-- 1.4

UPDATE PC SET nadmin = 2 WHERE numSerie = 2 AND numSerie = 1;

-- 1.5

UPDATE PC SET nadmin = NULL WHERE adresseIP = '193.54.227' AND numSerie = 2;

-- 1.6

DELETE FROM PC WHERE nadmin IS NULL;

--- Exercice 2

-- 2.1

CREATE TYPE elevage-type AS (
    typeanimal VARCHAR(30),
    ageMin INTEGER,
    nbrMax INTEGER
);

CREATE TABLE elevage OF elevage-type (
    PRIMARY KEY (typeanimal)
);

CREATE TYPE tadresse AS (
    numRue INTEGER,
    nomRue VARCHAR(30),
    ville VARCHAR(30),
    codePostal INTEGER
);

CREATE TYPE televeur AS (
    nLicence INTEGER,
    elevage elevage-type, 
    adresse tadresse
);

CREATE TABLE eleveurs OF televeur (
    PRIMARY KEY (nLicence),
    UNIQUE (elevage)
);

INSERT INTO elevage VALUES ('vache', 2, 10);
INSERT INTO elevage VALUES ('cheval', 3, 5);
INSERT INTO elevage VALUES ('bovin', 1, 20);
INSERT INTO elevage VALUES ('porcin', 1, 15);

INSERT INTO eleveurs VALUES (1, 'vache', 2, 10, 1, 'rue des vaches', 'Paris', 75000);
INSERT INTO eleveurs VALUES (2, 'cheval', 3, 5, 2, 'rue des chevaux', 'Paris', 75000);
INSERT INTO eleveurs VALUES (3, 'bovin', 1, 20, 3, 'rue des bovins', 'Paris', 75000);

-- 2.2

UPDATE eleveurs SET elevage = 'porcin' WHERE nLicence = 2;

-- 2.3

UPDATE eleveurs SET adresse.ville = 'Bordeaux' AND adresse.codePostal = 33000 WHERE elevage.typeanimal = 'bovin';

-- 2.4

UPDATE eleveurs SET elevage = NULL WHERE adresse.ville = 'Paris';

CREATE TRIGGER supprimer_elevage_parisien
BEFORE INSERT OR UPDATE ON eleveurs
FOR EACH ROW
BEGIN
    IF NEW.adresse.ville = 'Paris' THEN
        SET NEW.elevage = NULL;
    END IF;
END;

-- 2.5

UPDATE eleveurs SET elevage = 'volailles' WHERE adresse.ville = 'Angers';

CREATE TRIGGER eleveur_angevin
BEFORE INSERT OR UPDATE ON eleveurs
FOR EACH ROW
BEGIN
    IF NEW.adresse.ville = 'Angers' THEN
        SET NEW.elevage = 'volailles';
    END IF;
END;
