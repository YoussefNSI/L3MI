------------------------------------------------------------------
-- TD collection 

------------------  EXO 1
CREATE TYPE tposte AS OBJECT (
    nposte VARCHAR(5),
    typeposte VARCHAR(20)
);

CREATE TABLE poste OF tposte (
    PRIMARY KEY (nposte)
);

CREATE TYPE tlistpostes AS TABLE OF REF tposte;

CREATE TABLE segment (
    adrIP CHAR(9) PRIMARY KEY,
    Nomseg VARCHAR(20),
    longueur INT,
    listePoste tlistepostes,
    NESTED TABLE listePoste STORE AS temp1
);

INSERT INTO poste VALUES 
    ('P1', 'wind'),   
    ('P3', 'wind'),  
    ('P2', 'Linux');

INSERT INTO segment VALUES
    ('130.40.30', 'ICARE', 25, 
    tlistepostes(
        (SELECT REF(p) FROM poste p WHERE nposte = 'P1'), 
        (SELECT REF(p) FROM poste p WHERE nposte = 'P2'),
        (SELECT REF(p) FROM poste p WHERE nposte = 'P3')
    )
);

-- 1.1       
SELECT PS.nposte, PS.typeposte 
FROM TABLE (SELECT listePoste FROM segment WHERE Nomseg = 'Minos') PS;

-- 1.2
SELECT S.adrIP, S.longueur 
FROM segment S 
WHERE EXISTS (
    SELECT *  
    FROM TABLE (
        SELECT listePoste FROM segment S1 
        WHERE S.adrIP = S1.adrIP) PS
    WHERE PS.typeposte = 'windServ'
);

------------------  EXO 2

CREATE TYPE tprof AS OBJECT (
    numens VARCHAR(5),
    nomE VARCHAR(20),
    grade CHAR(2)
);

CREATE TABLE Enseig OF tprof (
    PRIMARY KEY (numens)
);

CREATE TYPE tform AS OBJECT (
    filiere VARCHAR(10),
    volume INT
);

CREATE TYPE tforms AS TABLE OF tform;

-- pas besoin de creer la table formation (volume ds une formation depend de la matiere)
CREATE TABLE Matiere ( 
    Titre VARCHAR(20), 
    Prof REF tprof,
    formations tforms,
    NESTED TABLE formations STORE AS temp
);

INSERT INTO Enseig VALUES 
    ('P1', 'ADELINE', 'PR'),   
    ('P2', 'NOAM', 'PR'),  
    ('P3', 'HANNA', 'MC');

INSERT INTO Matiere VALUES
    ('Algo',
    (SELECT REF(p) FROM Enseig p WHERE numens = 'P1'),
    tforms(
        tform('L3Info', 35),
        tform('L3PRO', 47),
        tform('L1MI', 57)
    )
);
