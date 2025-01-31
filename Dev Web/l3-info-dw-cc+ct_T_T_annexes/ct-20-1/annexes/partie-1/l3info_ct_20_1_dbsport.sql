-- phpMyAdmin SQL Dump
-- version 4.5.4.1deb2ubuntu2
-- http://www.phpmyadmin.net
--
-- Client :  localhost
-- Généré le :  Mer 10 Mars 2021 à 10:14
-- Version du serveur :  5.7.22-0ubuntu0.16.04.1
-- Version de PHP :  7.0.30-0ubuntu0.16.04.1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de données :  `dbsport`
--
CREATE DATABASE IF NOT EXISTS `l3info_ct_20_1_dbsport` DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
USE `l3info_ct_20_1_dbsport`;

-- --------------------------------------------------------

--
-- Structure de la table `championnats`
--

CREATE TABLE `championnats` (
  `id` bigint(20) NOT NULL,
  `nom` varchar(30) NOT NULL,
  `difficulte` tinyint(4) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Contenu de la table `championnats`
--

INSERT INTO `championnats` (`id`, `nom`, `difficulte`) VALUES
(1, 'DIVISION 1', 5),
(2, 'DIVISION 2', 4),
(3, 'DIVISION 3', 3);

-- --------------------------------------------------------

--
-- Structure de la table `equipes`
--

CREATE TABLE `equipes` (
  `id` bigint(20) NOT NULL,
  `nom` varchar(15) NOT NULL,
  `cmaillot` varchar(30) NOT NULL,
  `prestige` tinyint(4) NOT NULL,
  `idChampionnat` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Contenu de la table `equipes`
--

INSERT INTO `equipes` (`id`, `nom`, `cmaillot`, `prestige`, `idChampionnat`) VALUES
(1, 'ANGERS', 'BLANC', 8, 1),
(2, 'BARCELONE', 'ROUGE', 10, 1),
(3, 'LONDRES', 'VERT', 9, 1),
(4, 'UNIV', 'BLEU', 8, 2),
(5, 'MARSEILLE', 'BLANC', 6, 2),
(6, 'PARIS', 'ROUGE', 7, 2),
(7, 'MILAN', 'VERT', 5, 3),
(8, 'BERLIN', 'BLEU', 3, 3),
(9, 'MOSCOU', 'BLANC', 4, 3);

-- --------------------------------------------------------

--
-- Structure de la table `joueurs`
--

CREATE TABLE `joueurs` (
  `id` bigint(20) NOT NULL,
  `nlicence` varchar(15) NOT NULL,
  `nom` varchar(30) NOT NULL,
  `prenom` varchar(30) NOT NULL,
  `buts` bigint(20) NOT NULL,
  `idEquipe` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Contenu de la table `joueurs`
--

INSERT INTO `joueurs` (`id`, `nlicence`, `nom`, `prenom`, `buts`, `idEquipe`) VALUES
(1, '785463287448777', 'TORRES', 'CRISTIANO', 3, 1),
(2, '635489721012345', 'KARABATIC', 'NIKOLA', 12, 1),
(3, '789421036874987', 'DJOKOVIC', 'ROGER', 6, 2),
(4, '478965487321548', 'DESCHAMPS', 'DIDIER', 20, 3),
(5, '357984621036875', 'NORRIS', 'CHUCK', 36, 4),
(6, '456307894561784', 'LAMA', 'BERNARD', 9, 4),
(7, '657810325498684', 'BARTHEZ', 'FABIEN', 20, 5),
(8, '413569874563217', 'ABALO', 'LUC', 16, 6),
(9, '986321547890324', 'OMEYER', 'THIERRY', 3, 6),
(10, '654781249354784', 'BLANC', 'MICHEL', 4, 7),
(11, '523014789654123', 'PUYOL', 'CARLES', 14, 8),
(12, '412579863012798', 'TYSON', 'MIKE', 30, 8),
(13, '412369875469787', 'CURRY', 'STEPHEN', 40, 9),
(14, '478965412354789', 'JAMES', 'LEBRON', 20, 9);

--
-- Index pour les tables exportées
--

--
-- Index pour la table `championnats`
--
ALTER TABLE `championnats`
  ADD PRIMARY KEY (`id`);

--
-- Index pour la table `equipes`
--
ALTER TABLE `equipes`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `nom` (`nom`),
  ADD KEY `idChampionnat` (`idChampionnat`);

--
-- Index pour la table `joueurs`
--
ALTER TABLE `joueurs`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `nlicence` (`nlicence`),
  ADD KEY `idEquipe` (`idEquipe`);

--
-- AUTO_INCREMENT pour les tables exportées
--

--
-- AUTO_INCREMENT pour la table `championnats`
--
ALTER TABLE `championnats`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
--
-- AUTO_INCREMENT pour la table `equipes`
--
ALTER TABLE `equipes`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=17;
--
-- AUTO_INCREMENT pour la table `joueurs`
--
ALTER TABLE `joueurs`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
