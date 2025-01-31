<?php

function random_color()
{
  $r = dechex(rand(0, 255));
  $g = dechex(rand(0, 255));
  $b = dechex(rand(0, 255));
  return "#$r$g$b";
}

?>