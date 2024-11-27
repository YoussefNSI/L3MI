# Espaces de noms et autochargement avec `composer`. Tests unitaires avec `PHPUnit`.

1. Créez le fichier `composer.json` avec ce fragment pour 
	- pouvoir utiliser l'autochargeur PSR-4  de `composer` qui associera l'espace de noms (alias NS) dénommé `Acme` au  répertoire `src`
	- installer `PHPUnit` version 10.

```json
    "require-dev": {
        "phpunit/phpunit": "^10"
    },
	"autoload" : {
		"psr-4" : {
			"Acme\\" : "src/"
		}
	}
```

2. Installez l'autochargeur (`./vendor/autoload.php`) et `PHPUnit` avec `composer` (Note. La commande est à exécuter en dehors du conteneur. Les options `--ignore-` sont nécessaires sous les PC AIO du fait d'extensions manquantes dans la distribution `PHP (CLI)` qui y est installée.):

```sh
$ php composer.phar install --ignore-platform-req=ext-dom  --ignore-platform-req=ext-mbstring --ignore-platform-req=ext-xml --ignore-platform-req=ext-xmlwriter --ignore-platform-req=php
```


3. Définissez vos classes et vos interfaces dans des fichiers séparés portant le même nom que la classe ou l'interface (p. ex. `Employee.php` pour la classe `Employee`). Les classes de tests (ici une seule classe `ManagerTest`) doivent impérativement se terminer par `Test` pour être exécutées automatiquement par `PHPUnit`.

4. Créez les répertoires `src` et `tests` et déplacez le fichier de tests `ManagerTest.php` dans `tests` et les autres dans `src`. Mettez à jour l'autochargeur :

```sh
$ php composer.phar dump-autoload
```


5. Supprimez tous les imports de fichiers (`include`, `require`) figurant dans les fichiers de `src`, les remplacer par la déclaration du NS `Acme` et importez l'autochargeur, p. ex.

```php
<?php // ./src/Employee.php
declare(strict_types=1); // lève une exception si erreur de typage à l'appel de fonctions/méthodes
namespace Acme;
require_once __DIR__ . '/../vendor/autoload.php';

```

6. Implémentez vos scripts (p. ex. `./src/employee_display.php`) en y important les classes/interfaces requises par utilisation de leur FQCN (Fully Qualified Class Name) avec le mot-clé `use` et en important l'autochargeur, p. ex.

```php
<?php // ./src/employee_dislay.php
declare(strict_types=1);
require_once __DIR__ . '/../vendor/autoload.php';
use Acme\Employee;
```

7. Implémentez la classe de tests `tests/ManagerTest.php`  en y important les NS requis et l'autochargeur, p. ex.

```php
<?php // ./tests/ManagerTest.php
require_once __DIR__ . '/../vendor/autoload.php';
use Acme\Manager;
use Acme\Employee;
use PHPUnit\Framework\TestCase;

final class ManagerTest extends TestCase { ...
```
	
8. Une fois la classe de tests implantée, exécutez vos tests avec `PHPUnit` :

```sh
$ ./vendor/bin/phpunit tests
PHPUnit 10.4.2 by Sebastian Bergmann and contributors.

Runtime:       PHP 8.2.11

........R                                                           9 / 9 (100%)

Time: 00:00.009, Memory: 8.00 MB

There was 1 risky test:

1) ManagerTest::testAddEmployee
This test did not perform any assertions

/Users/david.lesaint/Documents/esg/l3info-dw/wsp/tp-php-2/corrections/tests/ManagerTest.php:41

OK, but there were issues!
Tests: 9, Assertions: 8, Risky: 1.
``` 
 

