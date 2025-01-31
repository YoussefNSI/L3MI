-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Hôte : localhost
-- Généré le : jeu. 05 déc. 2024 à 10:03
-- Version du serveur : 11.6.2-MariaDB
-- Version de PHP : 8.4.1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de données : `l3_cc_24_php_rush_hour`
--
CREATE DATABASE IF NOT EXISTS `l3_cc_24_php_rush_hour` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `l3_cc_24_php_rush_hour`;

-- --------------------------------------------------------

--
-- Structure de la table `DEFI`
--

CREATE TABLE `DEFI` (
  `defi_id` int(11) NOT NULL,
  `nr_cases` int(11) NOT NULL,
  `nr_voitures` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- RELATIONS POUR LA TABLE `DEFI`:
--

--
-- Déchargement des données de la table `DEFI`
--

INSERT INTO `DEFI` (`defi_id`, `nr_cases`, `nr_voitures`) VALUES
(1, 6, 4),
(2, 6, 4),
(3, 6, 6);

-- --------------------------------------------------------

--
-- Structure de la table `MOUVEMENT`
--

CREATE TABLE `MOUVEMENT` (
  `mouvement_id` int(11) NOT NULL,
  `partie_id` int(11) NOT NULL,
  `voiture_id` int(11) NOT NULL,
  `decalage` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- RELATIONS POUR LA TABLE `MOUVEMENT`:
--   `partie_id`
--       `PARTIE` -> `partie_id`
--   `voiture_id`
--       `VOITURE` -> `voiture_id`
--

-- --------------------------------------------------------

--
-- Structure de la table `PARTIE`
--

CREATE TABLE `PARTIE` (
  `partie_id` int(11) NOT NULL,
  `defi_id` int(11) NOT NULL,
  `debut` timestamp NULL DEFAULT NULL,
  `fin` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- RELATIONS POUR LA TABLE `PARTIE`:
--   `defi_id`
--       `DEFI` -> `defi_id`
--

-- --------------------------------------------------------

--
-- Structure de la table `POSITION`
--

CREATE TABLE `POSITION` (
  `defi_id` int(11) NOT NULL,
  `voiture_id` int(11) NOT NULL,
  `position` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- RELATIONS POUR LA TABLE `POSITION`:
--   `defi_id`
--       `DEFI` -> `defi_id`
--   `voiture_id`
--       `VOITURE` -> `voiture_id`
--

--
-- Déchargement des données de la table `POSITION`
--

INSERT INTO `POSITION` (`defi_id`, `voiture_id`, `position`) VALUES
(1, 1, 14),
(1, 2, 28),
(1, 5, 21),
(1, 9, 4),
(2, 1, 13),
(2, 5, 22),
(2, 6, 3),
(2, 9, 10),
(3, 1, 13),
(3, 2, 23),
(3, 3, 16),
(3, 5, 1),
(3, 7, 6),
(3, 8, 5);

-- --------------------------------------------------------

--
-- Structure de la table `VOITURE`
--

CREATE TABLE `VOITURE` (
  `voiture_id` int(11) NOT NULL,
  `couleur` varchar(15) NOT NULL,
  `taille` int(11) NOT NULL,
  `orientation` enum('H','V') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- RELATIONS POUR LA TABLE `VOITURE`:
--

--
-- Déchargement des données de la table `VOITURE`
--

INSERT INTO `VOITURE` (`voiture_id`, `couleur`, `taille`, `orientation`) VALUES
(1, 'red', 2, 'H'),
(2, 'cyan', 2, 'H'),
(3, 'cyan', 2, 'V'),
(4, 'orange', 2, 'H'),
(5, 'orange', 2, 'V'),
(6, 'blue', 2, 'H'),
(7, 'blue', 2, 'V'),
(8, 'purple', 2, 'H'),
(9, 'purple', 2, 'V'),
(10, 'pink', 2, 'H'),
(11, 'pink', 2, 'V'),
(12, 'green', 2, 'H'),
(13, 'green', 2, 'V'),
(14, 'yellow', 3, 'H'),
(15, 'yellow', 3, 'H'),
(16, 'grey', 2, 'H'),
(17, 'grey', 2, 'H'),
(18, 'beige', 2, 'H'),
(19, 'beige', 2, 'H');

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `DEFI`
--
ALTER TABLE `DEFI`
  ADD PRIMARY KEY (`defi_id`);

--
-- Index pour la table `MOUVEMENT`
--
ALTER TABLE `MOUVEMENT`
  ADD PRIMARY KEY (`mouvement_id`),
  ADD KEY `partie_id` (`partie_id`),
  ADD KEY `voiture_id` (`voiture_id`);

--
-- Index pour la table `PARTIE`
--
ALTER TABLE `PARTIE`
  ADD PRIMARY KEY (`partie_id`),
  ADD KEY `defi_id` (`defi_id`);

--
-- Index pour la table `POSITION`
--
ALTER TABLE `POSITION`
  ADD PRIMARY KEY (`defi_id`,`voiture_id`),
  ADD KEY `voiture_id` (`voiture_id`);

--
-- Index pour la table `VOITURE`
--
ALTER TABLE `VOITURE`
  ADD PRIMARY KEY (`voiture_id`);

--
-- AUTO_INCREMENT pour les tables déchargées
--

--
-- AUTO_INCREMENT pour la table `DEFI`
--
ALTER TABLE `DEFI`
  MODIFY `defi_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT pour la table `MOUVEMENT`
--
ALTER TABLE `MOUVEMENT`
  MODIFY `mouvement_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT pour la table `PARTIE`
--
ALTER TABLE `PARTIE`
  MODIFY `partie_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT pour la table `VOITURE`
--
ALTER TABLE `VOITURE`
  MODIFY `voiture_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Contraintes pour les tables déchargées
--

--
-- Contraintes pour la table `MOUVEMENT`
--
ALTER TABLE `MOUVEMENT`
  ADD CONSTRAINT `mouvement_ibfk_1` FOREIGN KEY (`partie_id`) REFERENCES `PARTIE` (`partie_id`) ON UPDATE CASCADE,
  ADD CONSTRAINT `mouvement_ibfk_2` FOREIGN KEY (`voiture_id`) REFERENCES `VOITURE` (`voiture_id`) ON UPDATE CASCADE;

--
-- Contraintes pour la table `PARTIE`
--
ALTER TABLE `PARTIE`
  ADD CONSTRAINT `partie_ibfk_1` FOREIGN KEY (`defi_id`) REFERENCES `DEFI` (`defi_id`) ON UPDATE CASCADE;

--
-- Contraintes pour la table `POSITION`
--
ALTER TABLE `POSITION`
  ADD CONSTRAINT `position_ibfk_1` FOREIGN KEY (`defi_id`) REFERENCES `DEFI` (`defi_id`) ON UPDATE CASCADE,
  ADD CONSTRAINT `position_ibfk_2` FOREIGN KEY (`voiture_id`) REFERENCES `VOITURE` (`voiture_id`) ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
