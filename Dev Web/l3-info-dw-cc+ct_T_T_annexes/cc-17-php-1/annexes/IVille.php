<?php
interface IVille {
	public function setNom($nv);
	public function setRegion($nr);
	public function setPopulation($tv);
	public function setPrefecture();
	public function getNom();
	public function getRegion();
	public function getPopulation();
	public function isPrefecture();
	public function __toString();
}
?>
