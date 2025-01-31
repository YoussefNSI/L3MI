-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Hôte : localhost
-- Généré le : ven. 18 fév. 2022 à 19:14
-- Version du serveur : 10.4.21-MariaDB
-- Version de PHP : 8.0.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de données : `l3info_cc_21_php_biblio`
--
CREATE DATABASE IF NOT EXISTS `l3info_cc_21_php_biblio` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE `l3info_cc_21_php_biblio`;

-- --------------------------------------------------------

--
-- Structure de la table `auteurs`
--

CREATE TABLE `auteurs` (
  `id` int(11) NOT NULL COMMENT 'identifiant unique',
  `prenom` varchar(100) DEFAULT NULL COMMENT 'prénom de l''auteur',
  `nom` varchar(100) NOT NULL COMMENT 'nom de l''auteur',
  `ddn` char(4) DEFAULT NULL COMMENT 'Date De Naissance',
  `ddd` char(4) DEFAULT NULL COMMENT 'Date De Décès'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Déchargement des données de la table `auteurs`
--

INSERT INTO `auteurs` (`id`, `prenom`, `nom`, `ddn`, `ddd`) VALUES
(1, 'Dominique', 'A', '1968', ''),
(2, 'Ghada Abdel', 'Aal', '1978', ''),
(3, 'Héctor', 'Abad Faciolince', '1958', ''),
(4, 'Stephan', 'Abarbanell', '', ''),
(5, 'Saït Faïk', 'Abasiyanik', '1906', '1954'),
(6, 'Edward', 'Abbey', '1927', '1989'),
(7, '', 'Abd Al Malik', '1975', ''),
(8, 'Dima', 'Abdallah', '1977', ''),
(9, 'Kader', 'Abdolah', '1954', ''),
(10, 'Barzou', 'Abdourazzoqov', '1959', ''),
(11, 'Kazushige', 'Abe', '1968', ''),
(12, 'Kôbô', 'Abé', '1924', '1993'),
(13, 'Agnès', 'Abécassis', '1972', ''),
(14, 'Eliette', 'Abécassis', '1969', ''),
(15, 'Emmy', 'Abrahamson', '', ''),
(16, 'Alain', 'Absire', '', ''),
(17, 'Adrien', 'Absolu', '1982', ''),
(18, 'Anwar', 'Accawi', '1943', ''),
(19, 'Chantel', 'Acevedo', '', ''),
(20, 'Carole', 'Achache', '', ''),
(21, '', 'Acheng', '', ''),
(22, 'André', 'Aciman', '1951', ''),
(23, 'Elliot', 'Ackerman', '1980', ''),
(24, 'Peter', 'Ackroyd', '1949', ''),
(25, 'Renata', 'Ada-Ruata', '1948', ''),
(26, 'Claire', 'Adam', '', ''),
(27, 'Olivier', 'Adam', '1974', ''),
(28, 'Gabriela', 'Adamesteanu', '', ''),
(29, 'Ales', 'Adamovich', '', ''),
(30, 'Alice', 'Adams', '', ''),
(31, 'Carl', 'Aderhold', '', ''),
(32, 'Maylis', 'Adhémar', '1985', ''),
(33, 'Chimamanda Ngozi', 'Adichie', '1977', ''),
(34, 'Aravind', 'Adiga', '1974', ''),
(35, 'Kaouther', 'Adimi', '1986', ''),
(36, 'Nana Kwame', 'Adjei-Brenyah', '1991', ''),
(37, 'Camille', 'Adler', '', ''),
(38, 'Yassin', 'Adnan', '1970', ''),
(39, 'Ricardo', 'Adolfo', '1974', ''),
(40, 'Chris', 'Adrian', '', ''),
(41, 'Pierre', 'Adrian', '', ''),
(42, '', 'Affinity K', '', ''),
(43, 'Tatamkhulu', 'Afrika', '1920', '2002'),
(44, 'Simonetta', 'Agnello Hornby', '1945', ''),
(45, 'José Eduardo', 'Agualusa', '1960', ''),
(46, 'Milena', 'Agus', '1959', ''),
(47, 'José', 'Agustin', '', ''),
(48, 'Cecelia', 'Ahern', '1981', ''),
(49, 'Jean d\'', 'Aillon', '1948', ''),
(50, 'César', 'Aira', '1949', ''),
(51, 'Mohammed', 'Aïssaoui', '', ''),
(52, 'Tchinguiz', 'Aïtmatov', '1928', '2008'),
(53, 'Erlom', 'Akhvlediani', '1933', '2012'),
(54, 'Anne', 'Akrich', '1986', ''),
(55, 'Vasili Pavlovitch', 'Aksenov', '', ''),
(56, 'Ryûnosuke', 'Akutagawa', '1892', '1927'),
(57, '', 'Alain-Fournier', '1886', '1914'),
(58, 'Rabih', 'Alameddine', '1959', ''),
(59, 'Meryem', 'Alaoui', '', ''),
(60, 'Nelly', 'Alard', '', ''),
(61, 'Noga', 'Albalach', '', ''),
(62, 'Marie-Fleur', 'Albecker', '', ''),
(63, 'A.', 'Alberts', '', ''),
(64, 'Mitch', 'Albom', '1959', ''),
(65, 'Laura', 'Alcoba', '1968', ''),
(66, 'Kate', 'Alcott', '', ''),
(67, 'Louisa May', 'Alcott', '1832', '1888'),
(68, 'Tiit', 'Aleksejev', '1968', ''),
(69, 'Kangni', 'Alem', '', ''),
(70, 'Raja', 'Alem', '', ''),
(71, 'Vassilis', 'Alexakis', '1943', '2021'),
(72, 'Gaïa', 'Alexia', '', ''),
(73, 'Robert', 'Alexis', '1956', ''),
(74, 'Bakhtiar', 'Ali', '', ''),
(75, 'Jakuta', 'Alikavazovic', '1979', ''),
(76, 'Julien', 'Allaire', '', ''),
(77, 'Nina', 'Allan', '1966', ''),
(78, 'Catherine', 'Allégret', '', ''),
(79, 'Isabel', 'Allende', '1942', ''),
(80, 'Dorothy', 'Allison', '1949', ''),
(81, 'Selva', 'Almada', '1973', ''),
(82, 'Eugenia', 'Almeida', '1972', ''),
(83, 'Vincent', 'Almendros', '1978', ''),
(84, 'Isabelle', 'Alonso', '1953', ''),
(85, 'Taleb', 'Alrefai', '1958', ''),
(86, 'Andreas', 'Altmann', '1949', ''),
(87, 'Julia', 'Alvarez', '1950', ''),
(88, 'Jorge', 'Amado', '1912', '2001'),
(89, 'Djaïli', 'Amadou Amal', '', ''),
(90, '', 'Ambai', '1944', ''),
(91, 'Jean', 'Améry', '', ''),
(92, 'Jonathan', 'Ames', '', ''),
(93, 'Santiago Horacio', 'Amigorena', '1962', ''),
(94, 'Kingsley', 'Amis', '1922', '1995'),
(95, 'Martin', 'Amis', '1949', ''),
(96, 'Niccolo', 'Ammaniti', '1966', ''),
(97, 'Kebir Mustapha', 'Ammi', '1952', ''),
(98, 'Tahmima', 'Anam', '1975', ''),
(99, 'Laurie Halse', 'Anderson', '1961', ''),
(100, 'Lena', 'Andersson', '', '');

-- --------------------------------------------------------

--
-- Structure de la table `livres`
--

CREATE TABLE `livres` (
  `id` int(11) NOT NULL COMMENT 'identifiant unique du livre',
  `titre` varchar(500) NOT NULL COMMENT 'titre du livre',
  `ddp` char(4) DEFAULT NULL COMMENT 'Date De Parution',
  `id_auteur` int(11) NOT NULL COMMENT 'Identifiant unique de l''auteur'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Déchargement des données de la table `livres`
--

INSERT INTO `livres` (`id`, `titre`, `ddp`, `id_auteur`) VALUES
(1, 'Regarder l\'océan', '2015', 1),
(2, 'Cherche mari désespérément', '2019', 2),
(3, 'La Secrète', '2016', 3),
(4, 'Angosta : roman', '2010', 3),
(5, 'Le voyage de Lilya : roman', '2017', 4),
(6, 'Un homme inutile', '2007', 5),
(7, 'Seuls sont les indomptés : roman', '2015', 6),
(8, 'Méchantes blessures', '2019', 7),
(9, 'Mauvaises herbes', '2020', 8),
(10, 'Le messager', '2018', 9),
(11, 'Huit monologues de femmes', '2013', 10),
(12, 'Sin semillas : roman', '2013', 11),
(13, 'L\'Homme-boîte', '1988', 12),
(14, 'La femme des sables', '2004', 12),
(15, 'Chouette, une ride !', '2010', 13),
(16, 'Le théorème de Cupidon : roman', '2012', 13),
(17, 'Au secours, il veut m\'épouser ! : roman', '2008', 13),
(18, 'Cherche ton bonheur partout', '2019', 13),
(19, 'Soirée sushi', '2011', 13),
(20, 'Week-end surprise : roman', '2015', 13),
(21, 'Mon père', '2002', 14),
(22, 'Mère et fille, un roman', '2008', 14),
(23, 'Un secret du docteur Freud', '2014', 14),
(24, 'La Répudiée', '2000', 14),
(25, 'Alyah', '2015', 14),
(26, 'La dernière tribu', '2006', 14),
(27, 'Clandestin', '2003', 14),
(28, 'Alyah', '2015', 14),
(29, 'Philothérapie', '2016', 14),
(30, 'Une affaire conjugale', '2010', 14),
(31, 'Nos rendez-vous : roman', '2020', 14),
(32, 'Mon père : roman', '2004', 14),
(33, 'Sépharade', '2011', 14),
(34, 'Comment tomber amoureuse d\'un homme qui vit dans un buisson', '2018', 15),
(35, 'Sans pays', '2007', 16),
(36, 'Mon sommeil sera paisible', '2014', 16),
(37, 'Les disparus du Joola', '2020', 17),
(38, 'L\'enfant de la Tour de la lune', '2010', 18),
(39, 'Lointaines merveilles', '2016', 19),
(40, 'Fille de', '2011', 20),
(41, 'Perdre son chemin', '1991', 21),
(42, 'Appelle-moi par ton nom : roman', '2018', 22),
(43, 'Le passage', '2020', 23),
(44, 'Trois frères', '2015', 24),
(45, 'L\'architecte assassin', '1990', 24),
(46, 'Battista revenait au printemps', '2014', 25),
(47, 'L\'enfant en or', '2019', 26),
(48, 'A l\'abri de rien', '2007', 27),
(49, 'Passer l\'hiver', '2004', 27),
(50, 'Je vais bien, ne t\'en fais pas', '2011', 27),
(51, 'La renverse', '2016', 27),
(52, 'La renverse', '2016', 27),
(53, 'La tête sous l\'eau', '2018', 27),
(54, 'Chanson de la ville silencieuse', '2018', 27),
(55, 'Une partie de badminton', '2019', 27),
(56, 'Des vents contraires', '2009', 27),
(57, 'Le coeur régulier : roman', '2011', 27),
(58, 'Falaises', '2006', 27),
(59, 'Falaises', '2006', 27),
(60, 'Une partie de badminton', '2019', 27),
(61, 'Tout peut s\'oublier', '2021', 27),
(62, 'Situation provisoire', '2013', 28),
(63, 'Viens et vois', '2015', 29),
(64, 'Un été invincible : roman', '2017', 30),
(65, 'Fermeture éclair : roman', '2012', 31),
(66, 'Le théâtre des nuits', '2020', 31),
(67, 'Bénie soit Sixtine', '2020', 32),
(68, 'Americanah', '2016', 33),
(69, 'L\'hibiscus pourpre', '2016', 33),
(70, 'La sélection', '2017', 34),
(71, 'Les ombres de Kittur', '2011', 34),
(72, 'Le tigre blanc', '2010', 34),
(73, 'Nos richesses', '2017', 35),
(74, 'Des pierres dans ma poche', '2016', 35),
(75, 'Les petits de Décembre', '2019', 35),
(76, 'Les petits de Décembre', '2019', 35),
(77, 'Friday black : nouvelles', '2021', 36),
(78, 'Le scénario parfait', '2015', 37),
(79, 'Hot Maroc', '2020', 38),
(80, 'Tout ce qui m\'est arrivé après ma mort', '2015', 39),
(81, 'Une nuit d\'été : roman', '2016', 40),
(82, 'Des âmes simples', '2017', 41),
(83, 'Les bons garçons', '2020', 41),
(84, 'Mischling', '2017', 42),
(85, 'Paradis amer', '2015', 43),
(86, 'L\'Amandière', '2003', 44),
(87, 'La Tante marquise', '2004', 44),
(88, 'Le secret de Torrenova', '2011', 44),
(89, 'La tante marquise', '2006', 44),
(90, 'La reine Ginga et comment les Africains ont inventé le monde', '2017', 45),
(91, 'Mon voisin', '2008', 46),
(92, 'Prends garde : le roman', '2015', 46),
(93, 'Sens dessus dessous', '2016', 46),
(94, 'Terres promises', '2018', 46),
(95, 'Battement d\'ailes', '2010', 46),
(96, 'La comtesse de Ricotta', '2013', 46),
(97, 'Une saison douce', '2021', 46),
(98, 'Une saison douce', '2021', 46),
(99, 'Mexico midi moins cinq', '1993', 47),
(100, 'Le joueur de billes', '2017', 48),
(101, 'L\'année où je t\'ai rencontré', '2019', 48),
(102, 'Juliette et les Cézanne', '2010', 49),
(103, 'La guerre des amoureuses', '2009', 49),
(104, 'La ville qui n\'aimait pas son roi', '2009', 49),
(105, 'La vie de Louis Fronsac : et autres nouvelles', '2013', 49),
(106, 'A lances et à pavois', '2020', 49),
(107, 'Varamo', '2005', 50),
(108, 'Prins', '2019', 50),
(109, 'Les funambules', '2020', 51),
(110, 'Djamilia', '2001', 52),
(111, 'Djamilia', '2003', 52),
(112, 'Un moustique dans la ville', '2017', 53),
(113, 'Il faut se méfier des hommes nus', '2017', 54),
(114, 'Traité de savoir-rire à l\'usage des embryons : roman', '2018', 54),
(115, 'Les hauts de Moscou : Moskva kva-kva', '2007', 55),
(116, 'La Vie d\'un idiot et autres nouvelles', '1987', 56),
(117, 'Colombe Blanchet : esquisses d\'un second roman inédit', '2003', 57),
(118, 'Le grand Meaulnes', '2009', 57),
(119, 'Les vies de papier', '2016', 58),
(120, 'L\'ange de l\'histoire', '2018', 58),
(121, 'La vérité sort de la bouche du cheval', '2018', 59),
(122, 'La vie que tu t\'étais imaginée', '2020', 60),
(123, 'Le vieil homme : des adieux', '2020', 61),
(124, 'Et j\'abattrai l\'arrogance des tyrans', '2018', 62),
(125, 'Iles', '2015', 63),
(126, 'Les Cinq personnes que j\'ai rencontrées là-haut', '2004', 64),
(127, 'Manèges : petite histoire argentine', '2015', 65),
(128, 'Jardin blanc : roman', '2009', 65),
(129, 'Si près des étoiles', '2020', 66),
(130, 'La petite couturière du Titanic : roman', '2019', 66),
(131, 'Pour le meilleur et pour le pire... et pour l\'éternité', '1996', 67),
(132, 'Le pèlerinage', '2018', 68),
(133, 'La légende de l\'assassin : roman', '2015', 69),
(134, 'Khâtem : une enfant d\'Arabie', '2011', 70),
(135, 'Je t\'oublierai tous les jours', '2005', 71),
(136, 'La clarinette', '2015', 71),
(137, 'Quand l\'homme de ses cauchemars ressurgit dans sa vie', '2019', 72),
(138, 'Quand la femme de sa vie resurgit de la nuit', '2019', 72),
(139, 'Mammon', '2011', 73),
(140, 'Le dernier grenadier du monde', '2019', 74),
(141, 'L\'avancée de la nuit', '2017', 75),
(142, 'La santé par l\'oeil : comprendre et utiliser l\'iridologie', '2018', 76),
(143, 'La fracture : roman', '2019', 77),
(144, 'Les pierres blanches', '2015', 78),
(145, 'Zorro', '2005', 79),
(146, 'Inès de mon âme', '2008', 79),
(147, 'Le jeu de Ripper : roman', '2015', 79),
(148, 'Portrait sépia : roman', '2007', 79),
(149, 'Inés de mon âme', '2010', 79),
(150, 'L\'amant japonais : roman', '2016', 79),
(151, 'La Cité des dieux sauvages', '2002', 79),
(152, 'Retour à Cayro', '2016', 80),
(153, 'Les jeunes mortes', '2015', 81),
(154, 'Après l\'orage', '2014', 81),
(155, 'Sous la grande roue', '2019', 81),
(156, 'La pièce du fond', '2010', 82),
(157, 'La pièce du fond', '2010', 82),
(158, 'Un été', '2015', 83),
(159, 'Ma chère Lise', '2011', 83),
(160, 'Faire mouche', '2018', 83),
(161, 'Je peux me passer de l\'aube', '2017', 84),
(162, 'Ici même : roman', '2015', 85),
(163, 'Al-Najdi, le marin', '2020', 85),
(164, 'La vie de merde de mon père, la vie de merde de ma mère et ma jeunesse de merde à moi : récit autobiographique', '2019', 86),
(165, 'Avec la mort en tenue de bataille', '2016', 87),
(166, 'Les Terres du bout du monde', '1991', 88),
(167, 'Capitaines des Sables', '1988', 88),
(168, 'Tieta d\'Agreste', '2007', 88),
(169, 'Tereza Batista : roman', '2011', 88),
(170, 'Les impatientes', '2020', 89),
(171, 'Les impatientes', '2020', 89),
(172, 'De haute lutte', '2015', 90),
(173, 'Les naufragés', '2010', 91),
(174, 'L\'homme de compagnie', '2011', 92),
(175, 'Le Premier Amour', '2004', 93),
(176, 'Le ghetto intérieur', '2019', 93),
(177, 'Le ghetto intérieur', '2019', 93),
(178, 'Les premières fois : roman', '2016', 93),
(179, 'Une adolescence taciturne : le second exil', '2002', 93),
(180, 'Une jeunesse aphone : les premiers arrangements', '2000', 93),
(181, 'Une enfance laconique', '1998', 93),
(182, 'Lucky Jim', '2014', 94),
(183, 'London fields', '1992', 95),
(184, 'La maison des rencontres', '2008', 95),
(185, 'La zone d\'intérêt', '2015', 95),
(186, 'Inside story', '2021', 95),
(187, 'Moi et toi', '2012', 96),
(188, 'La fête du siècle', '2011', 96),
(189, 'Mardochée : roman', '2011', 97),
(190, 'Les vaisseaux frères : roman', '2017', 98),
(191, 'Ma mémoire est un couteau', '2017', 99),
(192, 'Ester ou la passion pure : roman', '2015', 100);

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `auteurs`
--
ALTER TABLE `auteurs`
  ADD PRIMARY KEY (`id`);

--
-- Index pour la table `livres`
--
ALTER TABLE `livres`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_auteur` (`id_auteur`);

--
-- AUTO_INCREMENT pour les tables déchargées
--

--
-- AUTO_INCREMENT pour la table `auteurs`
--
ALTER TABLE `auteurs`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'identifiant unique', AUTO_INCREMENT=4846;

--
-- AUTO_INCREMENT pour la table `livres`
--
ALTER TABLE `livres`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'identifiant unique du livre', AUTO_INCREMENT=11481;

--
-- Contraintes pour les tables déchargées
--

--
-- Contraintes pour la table `livres`
--
ALTER TABLE `livres`
  ADD CONSTRAINT `livres_ibfk_1` FOREIGN KEY (`id_auteur`) REFERENCES `auteurs` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
