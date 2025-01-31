# TP PHP MVC

## Basé sur le [document](http://bpesquet.developpez.com/tutoriels/php/evoluer-architecture-mvc/) et le [dépôt](http://github.com/bpesquet/MonBlog) de [Baptiste Pesquet](https://github.com/bpesquet)


## Installation

- Installer `Twig 3.3`  avec `composer`

`[blog_mvc_objet]$ php composer.phar install` 

- Créer la base de données `l3_dw_tp_php_mvc_blog`

```
[blog_mvc_objet]$ mariadb -u etudiant -p
Enter password: 
...
MariaDB [(none)]> CREATE DATABASE IF NOT EXISTS l3_dw_tp_php_mvc_blog CHARACTER SET=utf8mb4 COLLATE utf8mb4_unicode_ci;
Query OK, 1 row affected (0.000 sec)
MariaDB [(none)]> exit;
Bye
[blog_mvc_objet]$ mariadb -u etudiant -p l3_dw_tp_php_mvc_blog < BD/l3_dw_tp_php_mvc_blog.sql 
Enter password: 
[blog_mvc_objet]$
```

- Ajuster la racine web dans le fichier `Config/dev.ini` à l'URI de votre répertoire

- Visiter la [page d'accueil](http://localhost/l3/tp-php-mvc/corrections/blog_mvc_objet/index.php)
