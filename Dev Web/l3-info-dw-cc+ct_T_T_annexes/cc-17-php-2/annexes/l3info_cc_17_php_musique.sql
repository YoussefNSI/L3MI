-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Hôte : localhost
-- Généré le : ven. 12 mars 2021 à 04:51
-- Version du serveur :  5.7.20
-- Version de PHP : 7.4.5

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de données : `l3info_cc_17_php_musique`
--

-- --------------------------------------------------------

--
-- Structure de la table `DISQUE`
--

CREATE TABLE `DISQUE` (
  `dsq_id` int(11) NOT NULL,
  `dsq_titre` varchar(100) COLLATE utf8mb4_bin NOT NULL,
  `dsq_annee` int(4) NOT NULL,
  `dsq_rang` int(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

--
-- Déchargement des données de la table `DISQUE`
--

INSERT INTO `DISQUE` (`dsq_id`, `dsq_titre`, `dsq_annee`, `dsq_rang`) VALUES
(25, 'Closer', 1980, 157),
(26, 'London Calling', 1979, 8),
(27, 'Hunky Dory', 1971, 107),
(28, 'Revolver', 1966, 2),
(29, 'Closer', 1980, 157),
(30, 'London Calling', 1979, 8),
(31, 'Hunky Dory', 1971, 107),
(32, 'Revolver', 1966, 2);

-- --------------------------------------------------------

--
-- Structure de la table `ENREGISTREMENT`
--

CREATE TABLE `ENREGISTREMENT` (
  `enr_id` int(11) NOT NULL,
  `enr_dsq` int(11) NOT NULL,
  `enr_mus` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

--
-- Déchargement des données de la table `ENREGISTREMENT`
--

INSERT INTO `ENREGISTREMENT` (`enr_id`, `enr_dsq`, `enr_mus`) VALUES
(8, 25, 43),
(9, 25, 44),
(10, 26, 45),
(11, 26, 46),
(12, 27, 47),
(13, 28, 48),
(14, 28, 49),
(15, 25, 43),
(16, 25, 44),
(17, 26, 45),
(18, 26, 46),
(19, 27, 47),
(20, 28, 48),
(21, 28, 49);

-- --------------------------------------------------------

--
-- Structure de la table `MUSICIEN`
--

CREATE TABLE `MUSICIEN` (
  `mus_id` int(11) NOT NULL,
  `mus_nom` varchar(20) COLLATE utf8mb4_bin NOT NULL,
  `mus_adn` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

--
-- Déchargement des données de la table `MUSICIEN`
--

INSERT INTO `MUSICIEN` (`mus_id`, `mus_nom`, `mus_adn`) VALUES
(43, 'Curtis', 1956),
(44, 'Summer', 1956),
(45, 'Jones', 1955),
(46, 'Strummer', 1952),
(47, 'Bowie', 1947),
(48, 'Lennon', 1940),
(49, 'McCartney', 1942);

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `DISQUE`
--
ALTER TABLE `DISQUE`
  ADD PRIMARY KEY (`dsq_id`);

--
-- Index pour la table `ENREGISTREMENT`
--
ALTER TABLE `ENREGISTREMENT`
  ADD PRIMARY KEY (`enr_id`);

--
-- Index pour la table `MUSICIEN`
--
ALTER TABLE `MUSICIEN`
  ADD PRIMARY KEY (`mus_id`);

--
-- AUTO_INCREMENT pour les tables déchargées
--

--
-- AUTO_INCREMENT pour la table `DISQUE`
--
ALTER TABLE `DISQUE`
  MODIFY `dsq_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=33;

--
-- AUTO_INCREMENT pour la table `ENREGISTREMENT`
--
ALTER TABLE `ENREGISTREMENT`
  MODIFY `enr_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=22;

--
-- AUTO_INCREMENT pour la table `MUSICIEN`
--
ALTER TABLE `MUSICIEN`
  MODIFY `mus_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=50;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
