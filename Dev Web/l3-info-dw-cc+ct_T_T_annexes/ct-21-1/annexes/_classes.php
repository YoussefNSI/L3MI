<?php
class Course {
    private $name;
    private $parts;
    public function __construct(string $name) {
        $this->name = $name;
    }
    public function getName() : string {
        return $this->name;
    }
    public function getParts() : array {
        return $this->parts;
    }
    public function setParts(array $parts) : void {
        $this->parts = $parts;
    }
}

class Part {
    private $name;
    private $groups;
    private $nrSessions;
    public function __construct(string $name, int $nrSessions) {
        $this->name = $name;
        $this->nrSessions = $nrSessions;
    }
    public function getName() : string {
        return $this->name;
    }
    public function getNrSessions() : int {
        return $this->nrSessions;
    }
    public function getGroups() : array {
        return $this->groups;
    }
    public function setGroups(array $groups) : void {
        $this->groups = $groups;
    }
}

class Group {
    private $name;
    private $maxHeadCount;
    public function __construct(string $name, int $maxHeadCount) {
        $this->name = $name;
        $this->maxHeadCount = $maxHeadCount;
    }
    public function getName() : string {
        return $this->name;
    }
    public function getMaxHeadCount() : int {
        return $this->maxHeadCount;
    }
}
?>