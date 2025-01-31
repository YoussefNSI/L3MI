-- phpMyAdmin SQL Dump
-- version 4.7.6
-- https://www.phpmyadmin.net/
--
-- Hôte : localhost
-- Généré le :  jeu. 01 fév. 2018 à 18:37
-- Version du serveur :  5.7.20
-- Version de PHP :  7.1.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de données :  `l3_tp_voitures`
--

-- --------------------------------------------------------

--
-- Structure de la table `cartegrise`
--

CREATE TABLE `cartegrise` (
  `id_pers` mediumint(5) NOT NULL DEFAULT '0',
  `immat` varchar(6) CHARACTER SET utf8 NOT NULL,
  `datecarte` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

-- --------------------------------------------------------

--
-- Structure de la table `modele`
--

CREATE TABLE `modele` (
  `id_modele` varchar(10) CHARACTER SET utf8 NOT NULL,
  `modele` varchar(30) CHARACTER SET utf8 NOT NULL,
  `carburant` enum('essence','diesel','gpl','électrique') CHARACTER SET utf8 NOT NULL DEFAULT 'essence'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `cartegrise`
--
ALTER TABLE `cartegrise`
  ADD PRIMARY KEY (`id_pers`,`immat`);

--
-- Index pour la table `modele`
--
ALTER TABLE `modele`
  ADD PRIMARY KEY (`id_modele`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
