-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Hôte : localhost
-- Généré le : ven. 12 mars 2021 à 04:52
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
-- Base de données : `l3info_cc_18_php_autoqcm`
--

-- --------------------------------------------------------

--
-- Structure de la table `ALTERNATIVE`
--

CREATE TABLE `ALTERNATIVE` (
  `ID_QUESTION` int(3) NOT NULL,
  `REPONSE` varchar(500) COLLATE utf8mb4_bin NOT NULL,
  `SOLUTION` tinyint(1) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

--
-- Déchargement des données de la table `ALTERNATIVE`
--

INSERT INTO `ALTERNATIVE` (`ID_QUESTION`, `REPONSE`, `SOLUTION`) VALUES
(1, 'Rien.', 0),
(1, 'Une erreur de niveau <code>E_WARNING</code> est émise.', 1),
(1, 'Une erreur fatale de niveau <code>E_COMPILE_ERROR</code> est émise.', 0),
(2, 'Rien.', 0),
(2, 'Une erreur de niveau <code>E_WARNING</code> est émise.', 0),
(2, 'Une erreur fatale de niveau <code>E_COMPILE_ERROR</code> est émise.', 1),
(3, '<code>NULL</code>', 1),
(3, '<code>boolean</code>', 0),
(3, '<code>undefined</code>', 0),
(4, '<code>boolean</code>', 0),
(4, '<code>integer</code>', 1),
(4, 'Une erreur de niveau <code>E_WARNING</code> est émise.', 0),
(5, '<code>bool(false)</code>', 0),
(5, '<code>bool(true)</code>', 1),
(6, '<code>bool(false)</code>', 1),
(6, '<code>bool(true)</code>', 0),
(7, '<code>#32FA62ZT</code>', 0),
(7, '<code>1</code>', 1),
(7, '<code>Undefined variable : $a</code>', 0),
(8, '<code>#32FA62ZT</code>', 0),
(8, '<code>2</code>', 1),
(8, '<code>Undefined variable : $a</code>', 0),
(9, '<code>bool(false)</code> ', 0),
(9, '<code>bool(true)</code>', 1),
(10, '<code>bool(false)</code> ', 0),
(10, '<code>bool(true)</code>', 1),
(11, '<code>-1</code>', 0),
(11, '<code>0</code>', 0),
(11, '<code>1</code> ', 1),
(12, '<code>-1</code>', 1),
(12, '<code>0</code>', 0),
(12, '<code>1</code> ', 0),
(13, '<code>bool(false)</code>', 1),
(13, '<code>bool(true)</code>', 0),
(14, '<code>bool(false)</code>', 0),
(14, '<code>int(0)</code>', 1),
(14, '<code>int(1)</code>', 0),
(15, '<code>Array()</code>', 0),
(15, '<code>Array([0] => 0 [1] => 0)</code>', 0),
(15, '<code>Array([0] => 0)</code>', 1),
(15, '<code>Array([1] => 0)</code>', 0),
(16, '<code>Array()</code>', 0),
(16, '<code>Array([0] => 0 [1] => 0)</code>', 0),
(16, '<code>Array([0] => 0)</code>', 1),
(16, '<code>Array([1] => 0)</code>', 0),
(17, '<code>Array ()</code> ', 0),
(17, '<code>Array( [0] => Array ( [0] => Array ()))</code>', 1),
(17, '<code>Array([0] => [])</code>', 0),
(17, 'Une erreur de niveau <code>E_WARNING</code> est émise.', 0),
(18, '<code>Array ()</code> ', 0),
(18, '<code>Array( [0] => Array() [1] => Array ())</code>', 1),
(18, '<code>Array([0] => [])</code>', 0),
(18, 'Une erreur de niveau <code>E_WARNING</code> est émise.', 0),
(19, '<code>Array([1] => 0)</code>', 1),
(19, '<code>Array([1] => 1 [2] => 0)</code>', 0),
(19, '<code>Array([1] => 1)</code>', 0),
(20, '<code>Array( [0] => 1 [1] => 2)</code>', 0),
(20, '<code>Array( [0] => 2 [1] => 3)</code>', 1),
(20, '<code>Array( [1] => 2 [2] => 3)</code>', 0),
(20, '<code>Array([0] => [])</code>', 0),
(21, '<code>Array( 0 => 0 [1] => 1)</code>', 0),
(21, '<code>Array( [a] => 1 [b] => 0)</code>', 0),
(21, '<code>Array( [b] => 0 [a] => 1)</code>', 1),
(22, '<code>Array( 0 => 0 [1] => 1)</code>', 0),
(22, '<code>Array( 0 => 1 [1] => 0)</code>', 1),
(22, '<code>Array( [a] => 1 [b] => 0)</code>', 0),
(22, '<code>Array( [b] => 0 [a] => 1)</code>', 0),
(23, '<code>Array( [0] => \'a\' [1] => 0 [a] => 0 )</code>', 0),
(23, '<code>Array( [0] => \'a\' [1] => 0)</code>', 0),
(23, '<code>Array( [0] => 1 [1] => 0 [a] => 0 )</code>', 1),
(23, '<code>Array( [0] => 1 [a] => 0)</code> ', 0),
(24, '<code>Array( [0] => 0 [1] => 0 [-1] => 0 )</code>', 0),
(24, '<code>Array( [0] => 1 [1] => 0)</code>', 0),
(24, '<code>Array( [0] => 1 [1] => 1 [-1] => 0 )</code>', 1),
(24, '<code>Array( [0] => 1 [2] => 0)</code>', 0),
(25, '<code>$x</code>', 1),
(25, '<code>\'1</code>', 0),
(25, '<code>\\\'1</code>', 0),
(25, 'Une erreur fatale est émise.', 0),
(26, '<code>\"1</code>', 0),
(26, '<code>$x</code>', 0),
(26, '<code>\'\"1\'</code>', 0),
(26, '<code>\\\"1</code>', 1),
(26, 'Une erreur fatale est émise.', 0),
(27, '<code>$tab[$i]</code>', 0),
(27, '<code>$tab[1]</code>', 0),
(27, '<code>1</code>', 0),
(27, '<code>2</code>', 1),
(28, '<code>$tab[$i] :2</code>', 0),
(28, '<code>$tab[1] :2</code>', 0),
(28, '<code>1 :2</code>', 1),
(29, '<code>1&lt;=&quot;2&quot;</code> ', 1),
(29, '<code>1<=\"2\"</code>', 0),
(29, '<code>1\\<=\"2\"</code>', 0),
(29, '<code>1\\<=\\\"2\\\"</code>', 0),
(30, '<code>ba</code>', 0),
(30, '<code>cde</code> ', 1),
(30, '<code>ef</code>', 0),
(31, '<code>0</code>', 1),
(31, '<code>1</code>', 0),
(32, '<code>hemingway at blue dot ocean</code> ', 1),
(32, '<code>hemingway at blue.ocean</code>', 0),
(32, '<code>hemingway@blue.ocean</code>', 0),
(33, '<code>21</code>', 0),
(33, '<code>3</code>', 0),
(33, '<code>Notice: Undefined variable : a</code>', 1),
(34, '<code>2</code> ', 0),
(34, '<code>3</code>', 0),
(34, '<code>Notice: Undefined index : a</code>', 1),
(35, '<code>JS ${JS}</code>', 0),
(35, '<code>JS ?</code>', 1),
(35, '<code>JS pourquoi pas</code>', 0),
(35, '<code>pourquoi pas JS</code>', 0),
(36, '<code>Et ${Et}</code>', 0),
(36, '<code>Et ?</code>', 1),
(36, '<code>Et pourquoi pas</code>', 0),
(36, '<code>pourquoi pas Et</code>', 0),
(37, '<code>-;-cesar;-paxa-romana</code> ', 1),
(37, '<code>;-cesar;-paxa-romana</code>', 0),
(37, '<code>Array( [1] => ; [2] => cesar; [3] => paxa [4] => romana)</code>', 0),
(38, '<code>Array( [0] => : [1] => :A [2] => :B:C)</code>', 0),
(38, '<code>Array( [0] => :: [1] => :A: [2] => ::B::C::)</code>', 0),
(38, '<code>Array( [0] => [1] => :A [2] => :B:C)</code>', 1),
(39, '<code>Array( [-1] => 2 [1] => -2)</code>', 0),
(39, '<code>Array( [-1] => 2 [1] => 2)</code>', 1),
(39, '<code>Array( [1] => 2)</code>', 0),
(40, '<code>Array( [-1] => -2 [1] => 2)</code>', 1),
(40, '<code>Array( [-1] => -2)</code>', 0),
(40, '<code>Array( [-1] => 2 [1] => -2)</code>', 0),
(40, '<code>Array( [1] => 2)</code>', 0);

-- --------------------------------------------------------

--
-- Structure de la table `QUESTION`
--

CREATE TABLE `QUESTION` (
  `ID` int(3) NOT NULL,
  `ENONCE` varchar(500) COLLATE utf8mb4_bin NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

--
-- Déchargement des données de la table `QUESTION`
--

INSERT INTO `QUESTION` (`ID`, `ENONCE`) VALUES
(1, 'Que fait l\'instruction <code>include(\'file.php\');</code> si <code>file.php</code> n\'existe pas ?'),
(2, 'Que fait l\'instruction <code>require(\'file.php\');</code> si <code>file.php</code> n\'existe pas ?'),
(3, '<code>$x = null; echo gettype($x);</code>'),
(4, '<code>echo gettype($x=2);</code>'),
(5, '<code>$x =\'\'; var_dump(empty($x));</code>'),
(6, '<code>$x; var_dump(isset($x));</code>'),
(7, '<code>$b=1; $a=&$b; unset($b); echo $a;</code>'),
(8, '<code>$b=2; $c=&$b; $a=&$c; unset($c); echo $a;</code>'),
(9, '<code>var_dump(TRUE!==1);</code>'),
(10, '<code>var_dump(TRUE==1);</code>'),
(11, '<code>echo 234 <=> 123;</code>'),
(12, '<code>echo \'aA1\' <=> \'1Aa\';</code>'),
(13, '<code>$x=false; $y=true; var_dump($x ?? $y);</code>'),
(14, '<code>$x=0; $y=true; var_dump($x ?? $y);</code>'),
(15, '<code>$tab[]=0; $tab[]=array_pop($tab); print_r($tab);</code>'),
(16, '<code>$tab[]=0; $tab[]=array_shift($tab); print_r($tab);</code>'),
(17, '<code>$tab[]=[]; array_push($tab[0],$tab[0]); print_r($tab);</code>'),
(18, '<code>$tab[]=[]; array_unshift($tab,$tab[0]); print_r($tab);</code>'),
(19, '<code>$tab[1]=1; $tab[count($tab)]=0; print_r($tab);</code>'),
(20, '<code>print_r(array_slice([1,2,3,4],1,2));</code>'),
(21, '<code>$tab=[\'a\'=>1,\'b\'=>0]; asort($tab); print_r($tab);</code>'),
(22, '<code>$tab=[\'a\'=>1,\'b\'=>0]; rsort($tab); print_r($tab);</code>'),
(23, '<code>$tab=[\'a\',0]; foreach($tab as $k=>$v){$tab[$v]=$k;} print_r($tab);</code>'),
(24, '<code>$tab=[1,0]; foreach($tab as $k=>$v){$tab[$k-$v]=$k;} print_r($tab);</code>'),
(25, '<code>$x=\'\\\'1\'; echo \'$x\';</code>'),
(26, '<code>$x=\'\\\"1\'; echo \"$x\";</code>'),
(27, '<code>$tab=[1,2]; $i=1; echo \"$tab[$i]\";}'),
(28, '<code>$tab=[1,2]; $i=1; echo \"{$tab[$i]} : \", $tab[$i];</code>'),
(29, '<code>echo htmlentities(\'1<=\"2\"\');</code>'),
(30, '<code>echo substr(\"abcdef\",2,-1);</code>'),
(31, '<code>echo preg_match(\'/\\^{}[A-Z]?-[0-9]+$/\', \"FR-49000\");</code>'),
(32, '<code>echo preg_replace(\'/([a-z]+)@([\\^{}.]+)\\.([a-z]+)/\',\"$1 at $2 dot $3\",\"hemingway@blue.ocean\");</code>'),
(33, '<code>$a=\"2\"; function test() {return $a+1;} echo test();</code>'),
(34, '<code>function test() { $a=2; return $GLOBALS[\'a\']+1; } echo test();</code>'),
(35, '<code>$js=\"JS\"; $$js=\"pourquoi pas\"; $JS=\"?\"; echo \"$js ${$js}\";</code>'),
(36, '<code>$et=\"Et\"; $$et=\"pourquoi pas\"; $Et=\"?\"; echo \"$et ${$et}\";</code>'),
(37, '<code>echo implode(\"-\",(explode(\":\",\":;:cesar;:paxa:romana\")));</code>'),
(38, '<code>print_r(explode(\";\",(implode(\":\",[\';\',\'A;\',\'B\',\'C\']))));</code>'),
(39, '<code>$t=[-1=>2,1=>-2]; print_r(array_map(\"abs\",$t));</code>'),
(40, '<code>$t=[-1=>2,1=>-2]; print_r(array_map(function($x){return -$x;},$t));</code>');

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `ALTERNATIVE`
--
ALTER TABLE `ALTERNATIVE`
  ADD PRIMARY KEY (`ID_QUESTION`,`REPONSE`);

--
-- Index pour la table `QUESTION`
--
ALTER TABLE `QUESTION`
  ADD PRIMARY KEY (`ID`);

--
-- AUTO_INCREMENT pour les tables déchargées
--

--
-- AUTO_INCREMENT pour la table `QUESTION`
--
ALTER TABLE `QUESTION`
  MODIFY `ID` int(3) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=41;

--
-- Contraintes pour les tables déchargées
--

--
-- Contraintes pour la table `ALTERNATIVE`
--
ALTER TABLE `ALTERNATIVE`
  ADD CONSTRAINT `alternative_ibfk_1` FOREIGN KEY (`ID_QUESTION`) REFERENCES `question` (`ID`) ON DELETE CASCADE ON UPDATE NO ACTION;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
