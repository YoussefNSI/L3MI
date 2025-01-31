<?php

/*
 * convertit les caractères spéciaux de la chaîne argument en entités HTML en laissant
 * inchangées les sous-chaînes <code> et </code> et renvoie la chaîne résultat.
 *
 * Exemples d'exécution :
 *
 * Que fait l'instruction <code>include('file.php');</code> si <code>file.php</code> n'existe pas ?
 * =>
 * Que fait l'instruction <code>include('file.php');</code> si <code>file.php</code> n'existe pas ?
 *
 * <code>echo 234 <=> 123;</code>
 * =>
 * <code>echo 234 &lt;=&gt; 123;</code>
 *
 * <code>$x='\"1'; echo "$x";</code>
 * =>
 * <code>$x='\&quot;1'; echo &quot;$x&quot;;</code>
 *
 * @param string $enonce La chaîne à traiter.
 *
 * @return string La chaîne obtenue après traitement.
 */
function encoderEnonce($enonce)
{
    // TODO
}
?>