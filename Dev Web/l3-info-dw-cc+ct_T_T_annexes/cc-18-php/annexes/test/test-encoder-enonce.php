<?php
require __DIR__ . '/../src/encoder-enonce.php';

$enonces = [];
$enonces[] = 'Que fait l\'instruction <code>include(\'file.php\');</code> si <code>file.php</code> n\'existe pas ?';
$enonces[] = '<code>echo 234 <=> 123;</code>';
$enonces[] = '<code>$x=\'\\\"1\'; echo \"$x\";</code>';

foreach ($enonces as $enonce) {
    echo $enonce . "\n";
    echo encoderEnonce($enonce) . "\n";
}
?>