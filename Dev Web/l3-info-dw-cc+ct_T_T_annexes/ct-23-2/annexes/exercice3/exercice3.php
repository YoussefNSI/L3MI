<?php

// PHP: SESSION/COOKIES
//
// Exemple du code pour rediriger ver l'ENT de l'Université d'Angers:
//  header("Location: https://ent.univ-angers.fr/");
//  exit;
//

// A COMPLETER

?>

<!DOCTYPE html>
<html lang="fr">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="../utils/style.css">
  <style>
    #datetime-bar {
      background-color: <?php /* A COMPLETER*/ ?>;
      color: white;
    }
  </style>
  <title>CT 2023 JS/PHP</title>
</head>

<body>
  <header id="datetime-bar">
    <p>Aujourd'hui: <?php echo date('l j F Y') ?></p>
  </header>

  <section>
    <h2>Service d'affichage de cartes cognitives</h2>
    <form method="post" action="#">
      <label for="map">Choisissez une carte cognitive:
        <br>
        <select name="map" id="map">
          <option value="1">Map_1</option>
          <option value="2">Map_2</option>
        </select>
      </label>
      <input type="submit" name="submit" value="Afficher">
    </form>
  </section>

  <section id="result">
    <table>
      <thead>
        <tr>
          <th></th>
          <th>n1</th>
          <th>n2</th>
          <th>n3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>n1</th>
          <td></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>n2</th>
          <td>+</td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>n3</th>
          <td></td>
          <td>+</td>
          <td></td>
        </tr>
      </tbody>
    </table>
  </section>
  <hr>
  <footer>

  </footer>
</body>

</html>