-- phpMyAdmin SQL Dump
-- version 4.9.5deb2
-- https://www.phpmyadmin.net/
--
-- Hôte : localhost:3306
-- Généré le : mer. 30 mars 2022 à 02:51
-- Version du serveur :  8.0.28-0ubuntu0.20.04.3
-- Version de PHP : 7.4.3

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de données : `l3info_ct_21_1_courses`
CREATE DATABASE IF NOT EXISTS `l3info_ct_21_1_courses` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
USE `l3info_ct_21_1_courses`;
--

-- --------------------------------------------------------

--
-- Structure de la table `GROUPE`
--

CREATE TABLE `GROUPE` (
  `id` bigint UNSIGNED NOT NULL,
  `name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `maxHeadCount` int NOT NULL,
  `part` bigint UNSIGNED NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

--
-- Déchargement des données de la table `GROUPE`
--

INSERT INTO `GROUPE` (`id`, `name`, `maxHeadCount`, `part`) VALUES
(1, 'English-Les-1', 30, 1),
(2, 'English-Les-2', 30, 1),
(3, 'English-Les-3', 30, 1),
(4, 'English-Eval-1', 20, 2),
(5, 'Databases-Lec-1', 80, 3),
(6, 'Databases-Tut-1', 40, 4),
(7, 'Databases-Tut-2', 40, 4),
(8, 'Databases-Lab-1', 20, 5),
(9, 'Databases-Lab-2', 20, 5),
(10, 'Databases-Lab-3', 20, 5),
(11, 'Databases-LabEval-1', 20, 6),
(12, 'Databases-LabEval-2', 20, 6),
(13, 'Databases-LabEval-3', 20, 6),
(14, 'Web-Development-Lec-1', 80, 7),
(15, 'Web-Development-Lab-1', 25, 8),
(16, 'Web-Development-Lab-2', 28, 8),
(17, 'Web-Development-Lab-3', 25, 8),
(18, 'Web-Development-LabEval-1', 40, 9),
(19, 'Web-Development-LabEval-2', 40, 9),
(20, 'Web-Development-LabEval-3', 40, 9),
(21, 'Computer-Generated-Images-Lec-1', 80, 10),
(22, 'Computer-Generated-Images-Lab-1', 40, 11),
(23, 'Python-Les-1', 40, 12),
(24, 'Python-Lab-1', 20, 13),
(25, 'Python-Lab-2', 20, 13),
(26, 'Intelligent-Algorithms-Lec-1', 80, 14),
(27, 'Intelligent-Algorithms-Tut-1', 40, 15),
(28, 'Intelligent-Algorithms-Lab-1', 20, 16),
(29, 'Intelligent-Algorithms-Lab-2', 20, 16),
(30, 'Intelligent-Algorithms-Lab-3', 20, 16),
(31, 'Qt-Lec-1', 80, 17),
(32, 'Qt-Lab-1', 20, 18),
(33, 'Qt-Lab-2', 20, 18),
(34, 'Functional-programming-Lec-1', 80, 19),
(35, 'Functional-programming-Tut-1', 40, 20),
(36, 'Functional-programming-Tut-2', 40, 20),
(37, 'Functional-programming-Lab-1', 20, 21),
(38, 'Functional-programming-Lab-2', 20, 21),
(39, 'Functional-programming-Lab-3', 25, 21),
(40, 'Logic-programming-Lec-1', 80, 22),
(41, 'Logic-programming-Tut-1', 40, 23),
(42, 'Logic-programming-Tut-2', 40, 23),
(43, 'Logic-programming-Lab-1', 20, 24),
(44, 'Logic-programming-Lab-2', 20, 24),
(45, 'Logic-programming-Lab-3', 20, 24);

-- --------------------------------------------------------

--
-- Structure de la table `PARTIE`
--

CREATE TABLE `PARTIE` (
  `id` bigint UNSIGNED NOT NULL,
  `name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `nbSessions` int NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

--
-- Déchargement des données de la table `PARTIE`
--

INSERT INTO `PARTIE` (`id`, `name`, `nbSessions`) VALUES
(1, 'English-Les', 8),
(2, 'English-Eval', 1),
(3, 'Databases-Lec', 8),
(4, 'Databases-Tut', 9),
(5, 'Databases-Lab', 7),
(6, 'Databases-LabEval', 1),
(7, 'Web-Development-Lec', 12),
(8, 'Web-Development-Lab', 8),
(9, 'Web-Development-LabEval', 1),
(10, 'Computer-Generated-Images-Lec', 3),
(11, 'Computer-Generated-Images-Lab', 8),
(12, 'Python-Les', 5),
(13, 'Python-Lab', 4),
(14, 'Intelligent-Algorithms-Lec', 2),
(15, 'Intelligent-Algorithms-Tut', 2),
(16, 'Intelligent-Algorithms-Lab', 10),
(17, 'Qt-Lec', 4),
(18, 'Qt-Lab', 7),
(19, 'Functional-programming-Lec', 7),
(20, 'Functional-programming-Tut', 4),
(21, 'Functional-programming-Lab', 4),
(22, 'Logic-programming-Lec', 6),
(23, 'Logic-programming-Tut', 3),
(24, 'Logic-programming-Lab', 4);

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `GROUPE`
--
ALTER TABLE `GROUPE`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`),
  ADD KEY `PARTIE_FK` (`part`);

--
-- Index pour la table `PARTIE`
--
ALTER TABLE `PARTIE`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`);

--
-- AUTO_INCREMENT pour les tables déchargées
--

--
-- AUTO_INCREMENT pour la table `GROUPE`
--
ALTER TABLE `GROUPE`
  MODIFY `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=46;

--
-- AUTO_INCREMENT pour la table `PARTIE`
--
ALTER TABLE `PARTIE`
  MODIFY `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=25;

--
-- Contraintes pour les tables déchargées
--

--
-- Contraintes pour la table `GROUPE`
--
ALTER TABLE `GROUPE`
  ADD CONSTRAINT `PARTIE_FK` FOREIGN KEY (`part`) REFERENCES `PARTIE` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
