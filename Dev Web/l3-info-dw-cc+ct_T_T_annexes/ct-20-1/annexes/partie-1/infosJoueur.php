<?php

$nom= "Onyme";
$prenom= "Anne";
$nlicence= "123456789";
$buts= 8;
$indice=5;


$nequipe="LERIA";
$cmaillot="JAUNE";
$prestige=8;

$nChamp = "CHAMPIONS LEAGUE";
$difficulte = 10;
?>

<h2><?php echo "$prenom $nom";?></h2>
<h3>Championnat</h3>
<table>
  <tr>
    <th>Nom</th>
    <th>Difficulté</th>
  </tr>
  <tr>
    <td><?php echo $nChamp;?></td>
    <td><?php echo $difficulte;?></td>
  </tr>
</table>
<h3>Equipe</h3>
<table>
  <tr>
    <th>Nom</th>
    <th>Couleur maillot</th>
    <th>Prestige</th>
  </tr>
  <tr>
    <td><?php echo $nequipe;?></td>
    <td><?php echo $cmaillot;?></td>
    <td><?php echo $prestige;?></td>
  </tr>
</table>
<h3>Informations</h3>
<table>
  <tr>
    <th>Numéro licence</th>
    <th>Buts</th>
    <th>Indice de performance</th>
  </tr>
  <tr>
    <td><?php echo $nlicence;?></td>
    <td><?php echo $buts;?></td>
    <td><?php echo $indice;?></td>
  </tr>
</table>
<h3>Co-équipiers</h3>
<ul>
<li><a href="">Zizou</a></li>
</ul>
