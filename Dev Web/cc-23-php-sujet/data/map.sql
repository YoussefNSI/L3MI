-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Hôte : localhost
-- Généré le : mer. 22 nov. 2023 à 12:40
-- Version du serveur : 11.1.2-MariaDB
-- Version de PHP : 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";

USE mysql;


-- -------------------------------------------------------
-- Base de données : `l3_cc_23_php_map`
-- -------------------------------------------------------

DROP DATABASE IF EXISTS `l3_cc_23_php_map`;
CREATE DATABASE IF NOT EXISTS `l3_cc_23_php_map` CHARACTER SET `utf8mb4` COLLATE `utf8mb4_bin`;
USE `l3_cc_23_php_map`;


-- -------------------------------------------------------
-- Tables
-- -------------------------------------------------------

-- 
-- Table `MAP`
-- 
CREATE TABLE `MAP` (
  `ID`        int(11) NOT NULL,
  `NAME`      varchar(50) NOT NULL,
  `NUM_NODES` int(11) NOT NULL,
  `NUM_ARCS`  int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

INSERT INTO `MAP` (`ID`, `NAME`, `NUM_NODES`, `NUM_ARCS`) VALUES
(1, 'map_1', 3, 3);


--
-- Table `NODE`
--

CREATE TABLE `NODE` (
  `ID`       int(11) NOT NULL,
  `MAP`      int(11) NOT NULL,
  `NAME`     varchar(50) NOT NULL,
  `NODE_NUM` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

INSERT INTO `NODE` (`ID`, `MAP`, `NAME`, `NODE_NUM`) VALUES
(1, 1, 'n1', 1),
(2, 1, 'n2', 2),
(3, 1, 'n3', 3);


--
-- Table `ARC`
--

CREATE TABLE `ARC` (
  `ID`   int(11) NOT NULL,
  `TAIL` int(11) NOT NULL,
  `HEAD` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

INSERT INTO `ARC` (`ID`, `TAIL`, `HEAD`) VALUES
(1, 1, 2),
(2, 2, 3),
(3, 2, 3);


--
-- Table `INT_LABEL`
--

CREATE TABLE `INT_LABEL` (
  `ID`    int(11) NOT NULL,
  `VALUE` int(11) NOT NULL,
  `ARC`   int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;


--
-- Table `SYMBOLIC_LABEL`
--

CREATE TABLE `SYMBOLIC_LABEL` (
  `ID`    int(11) NOT NULL,
  `VALUE` enum('?','0','-','+') NOT NULL,
  `ARC`   int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;


INSERT INTO `SYMBOLIC_LABEL` (`ID`, `VALUE`, `ARC`) VALUES
(1, '+', 1),
(2, '+', 2),
(3, '+', 3);


-- -------------------------------------------------------
-- Index pour les tables déchargées
-- -------------------------------------------------------

ALTER TABLE `MAP`
  ADD PRIMARY KEY (`ID`);

ALTER TABLE `NODE`
  ADD PRIMARY KEY (`ID`),
  ADD KEY `fk_node_map` (`MAP`);

ALTER TABLE `ARC`
  ADD PRIMARY KEY (`ID`),
  ADD KEY `fk_arc_head` (`HEAD`),
  ADD KEY `fk_arc_tail` (`TAIL`);

ALTER TABLE `SYMBOLIC_LABEL`
  ADD PRIMARY KEY (`ID`),
  ADD KEY `fk_symb_label_arc` (`ARC`);

ALTER TABLE `INT_LABEL`
  ADD PRIMARY KEY (`ID`),
  ADD KEY `fk_int_label_arc` (`ARC`);



-- -------------------------------------------------------
-- AUTO_INCREMENT pour les tables déchargées
-- -------------------------------------------------------

ALTER TABLE `SYMBOLIC_LABEL`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT;

ALTER TABLE `INT_LABEL`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT;

ALTER TABLE `ARC` 
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT;

ALTER TABLE `NODE`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT;

ALTER TABLE `MAP`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT;


-- -------------------------------------------------------
-- Contraintes
-- -------------------------------------------------------

ALTER TABLE `NODE`
  ADD CONSTRAINT `fk_node_map` FOREIGN KEY (`MAP`) REFERENCES `MAP` (`ID`);

ALTER TABLE `ARC`
  ADD CONSTRAINT `fk_arc_node_head` FOREIGN KEY (`HEAD`) REFERENCES `NODE` (`ID`),
  ADD CONSTRAINT `fk_arc_node_tail` FOREIGN KEY (`TAIL`) REFERENCES `NODE` (`ID`);

ALTER TABLE `SYMBOLIC_LABEL`
  ADD CONSTRAINT `fk_s_label_arc` FOREIGN KEY (`ARC`) REFERENCES `ARC` (`ID`);

ALTER TABLE `INT_LABEL`
  ADD CONSTRAINT `fk_INT_LABEL_arc` FOREIGN KEY (`ARC`) REFERENCES `ARC` (`ID`);
