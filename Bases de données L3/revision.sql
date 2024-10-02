DROP TABLE IF EXISTS CLUB;
CREATE TABLE CLUB (
    NClub VARCHAR(50) UNIQUE,
    Region VARCHAR(100)
);

DROP SEQUENCE IF EXISTS numnag;
CREATE SEQUENCE numnag MINVALUE 101;
CREATE TABLE NAGEUR(
    NumeroNag INTEGER PRIMARY KEY,
    NomNag VARCHAR(50),
    PrenomNag VARCHAR(50),
    AnneedeNaissance INTEGER,
    Sexe CHAR(1) CHECK(Sexe IN ('M', 'F')),
    NomClub VARCHAR(50) REFERENCES CLUB(NClub)
);

DROP SEQUENCE IF EXISTS numep;
CREATE SEQUENCE numep;
DROP TABLE IF EXISTS CATEGORIE_EPREUVE;
CREATE TABLE CATEGORIE_EPREUVE(
    NumeroEp INTEGER PRIMARY KEY,
    Type CHAR(50) NOT NULL CHECK(Type IN ('Parcours à sec', 'Propulsion technique', 'Technique')),
    Niveau CHAR(50) NOT NULL CHECK(Niveau IN ('Synchro Découverte', 'Synchro Argent', 'Synchro Or'))
);

DROP SEQUENCE numoff;
CREATE SEQUENCE numoff;
DROP TABLE IF EXISTS OFFICIEL;
CREATE TABLE OFFICIEL (
    NumeroOff INTEGER PRIMARY KEY,
    NomOff VARCHAR(50),
    PrenomOff VARCHAR(50),
    Degre CHAR(1) CHECK(Degre IN ('A', 'B', 'C', 'D')) DEFAULT 'D',
    NomClub VARCHAR(50) REFERENCES CLUB(NClub)
);

DROP TABLE IF EXISTS RESULTAT;
CREATE TABLE RESULTAT (
    NumeroNag INTEGER REFERENCES NAGEUR(NumeroNag),
    NumeroEp INTEGER REFERENCES CATEGORIE_EPREUVE(NumeroEp),
    Annee INTEGER,
    NumeroOff INTEGER REFERENCES OFFICIEL(NumeroOff),
    Note NUMERIC(10),
    PRIMARY KEY(NumeroNag, NumeroEp, Annee, NumeroOff)
);

INSERT INTO CLUB VALUES('Angers Nat Synchro', 'Pays de la Loire');

INSERT INTO NAGEUR VALUES(nextval(numnag), 'ROBERT', 'Léna', 2006, 'F', 'Angers Nat Synchro');
INSERT INTO NAGEUR VALUES(nextval(numnag), 'LECOURT', 'Clément', 2008, 'M', 'Angers Nat Synchro');

INSERT INTO NAGEUR VALUES(nextval(numnag), 'CHAFFES', 'Lila', 2006, 'F', 'Angers Nat Synchro');
INSERT INTO CLUB VALUES('Leo Lagrange Nantes', 'Pays de la Loire');
INSERT INTO OFFICIEL(NumeroOff, NomOff, PrenomOff, NomClub) VALUES(nextval(numoff), 'BOZEC', 'Rachel', 'Leo Lagrange Nantes');
INSERT INTO CATEGORIE_EPREUVE VALUES(nextval(numep), 'Propulsion technique', 'Synchro Or');
INSERT INTO RESULTAT VALUES()
