<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Réponses QCM</title>
<style>
div {
	border: 1px solid black;
	margin: 2px;
	padding: 0px 5px 1px;
}

.correct {
	color: green;
}

.incorrect {
	color: red;
}
</style>
</head>
<body>
<?php
require 'encoder-enonce.php';
require 'connexpdo.inc.php';
require 'selectionner-questions-alternatives.php';

    // TODO
?>
<br />
	<a href="generer-qcm.php">Recommencer !</a>
	<a href="qcm.html">Accueil !</a>
</body>
</html>
