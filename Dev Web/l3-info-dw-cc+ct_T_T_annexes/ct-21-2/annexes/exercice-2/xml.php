<?php
// TODO 1/2
?>
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L3 Développement Web - CT2</title>
    <link rel="stylesheet" href="style.css">
</head>

<body>
    <h2>Comparateur de sources d'info</h2>
    <hr />
    <form method="post" action="xml.php">
        <div><label for="country">Pays&nbsp;&nbsp;<input type="text" id="country" name="country" value="<?= $countryfield ?>"></label>
            <button type="submit" name="source" value="xml">OK</button>
        </div>
        <label class="col1">Source XML<input type="radio" name="source" value="xml" checked /></label>
        <table class="col1" id="xml">
            <tr>
                <th>Pays</th>
                <th>Continent</th>
            </tr>
            <?php
            // TODO 2/2
            ?>
        </table>
    </form>
</body>

</html>