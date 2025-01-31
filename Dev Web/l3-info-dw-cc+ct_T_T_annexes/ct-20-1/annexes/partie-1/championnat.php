<?php include 'functions.php'?>
<h1>Modifier un championnat</h1>
<form action="<?php echo $_SERVER['PHP_SELF'];?>" name="formChampionnat"
	method="post">
<?php selectChampionnats();?>
	<br /><br /> <label><b>Difficulté : </b></label><input type="number"
		name="difficulte"
		value="<?php echo isset($difficulte) ? $difficulte: 0;?>" max="127" min="-128"><br />
	<br /> <input type="submit" name="setDifficulte" value="Modifier"><br />
	<br />
</form>