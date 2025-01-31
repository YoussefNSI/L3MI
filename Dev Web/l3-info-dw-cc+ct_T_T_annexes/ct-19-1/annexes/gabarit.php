<!DOCTYPE html>
<html>
<head>
<title>Countries</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" type="text/css" href="country.css" />
<script defer src="src/country-codes.js"></script>
<script defer src="src/country-continents.js"></script>
<script defer src="src/country-flags.js"></script>
<script defer src="utils-obf.js"></script>
<script defer src="country.js"></script>
</head>
<body>
	<!-- Header -->
	<div class="header"></div>
	<!-- Content -->
	<div class="row">
		<div class="side">
			<h2>Filter</h2>
			<label>Continent</label> <select name="continents" id="continents">
				<option value="all">*</option>
				<option value="Africa">Africa</option>
				<option value="Antarctica">Antarctica</option>
				<option value="Asia">Asia</option>
				<option value="Europe">Europe</option>
				<option value="North America">North America</option>
				<option value="Oceania">Oceania</option>
				<option value="South America">South America</option>
			</select> <br />
			<h2>Display</h2>
			<label>Names</label>
			<input type="radio" name="pays" value="noms" checked />
			<br />
			<label>Codes</label>
			<input type="radio" name="pays" value="codes" />
			<br />
			<label>Flags</label>
			<input type="radio" name="pays" value="drapeaux" />
			<br />
		</div>
		<div class="main">
			<table>
				<tbody>
<?php
// Q1-2 génération du tableau 16x16 des pays au format "<td id="France">France</td>"
?>
            </tbody>
			</table>
		</div>
	</div>
</body>
</html>
