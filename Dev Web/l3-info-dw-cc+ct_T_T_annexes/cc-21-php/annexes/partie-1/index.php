<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Examen sur Machine : Objets et Base de Données en PHP</title>
</head>
<body>
    <!-- Partie Afficher les livres  -->
    <form action="livresParAuteur.php" method="GET">
        <fieldset>
            <legend>Afficher les livres de l'auteur sélectionné</legend>
            <label for="auteur">Auteur : 
                <select name="auteur" id="auteur">
                    <option value="63">A. Alberts</option><option value="7"> Abd Al Malik</option><option value="21"> Acheng</option><option value="17">Adrien Absolu</option><option value="42"> Affinity K</option><option value="13">Agnès Abécassis</option><option value="57"> Alain-Fournier</option><option value="16">Alain Absire</option><option value="29">Ales Adamovich</option><option value="30">Alice Adams</option><option value="90"> Ambai</option><option value="86">Andreas Altmann</option><option value="22">André Aciman</option><option value="54">Anne Akrich</option><option value="18">Anwar Accawi</option><option value="34">Aravind Adiga</option><option value="74">Bakhtiar Ali</option><option value="10">Barzou Abdourazzoqov</option><option value="37">Camille Adler</option><option value="31">Carl Aderhold</option><option value="20">Carole Achache</option><option value="78">Catherine Allégret</option><option value="48">Cecelia Ahern</option><option value="19">Chantel Acevedo</option><option value="33">Chimamanda Ngozi Adichie</option><option value="40">Chris Adrian</option><option value="26">Claire Adam</option><option value="50">César Aira</option><option value="8">Dima Abdallah</option><option value="89">Djaïli Amadou Amal</option><option value="1">Dominique A</option><option value="80">Dorothy Allison</option><option value="6">Edward Abbey</option><option value="14">Eliette Abécassis</option><option value="23">Elliot Ackerman</option><option value="15">Emmy Abrahamson</option><option value="53">Erlom Akhvlediani</option><option value="82">Eugenia Almeida</option><option value="28">Gabriela Adamesteanu</option><option value="72">Gaïa Alexia</option><option value="2">Ghada Abdel Aal</option><option value="3">Héctor Abad Faciolince</option><option value="79">Isabel Allende</option><option value="84">Isabelle Alonso</option><option value="75">Jakuta Alikavazovic</option><option value="91">Jean Améry</option><option value="49">Jean d' Aillon</option><option value="92">Jonathan Ames</option><option value="88">Jorge Amado</option><option value="47">José Agustin</option><option value="45">José Eduardo Agualusa</option><option value="87">Julia Alvarez</option><option value="76">Julien Allaire</option><option value="9">Kader Abdolah</option><option value="69">Kangni Alem</option><option value="35">Kaouther Adimi</option><option value="66">Kate Alcott</option><option value="11">Kazushige Abe</option><option value="97">Kebir Mustapha Ammi</option><option value="94">Kingsley Amis</option><option value="12">Kôbô Abé</option><option value="65">Laura Alcoba</option><option value="99">Laurie Halse Anderson</option><option value="100">Lena Andersson</option><option value="67">Louisa May Alcott</option><option value="62">Marie-Fleur Albecker</option><option value="95">Martin Amis</option><option value="32">Maylis Adhémar</option><option value="59">Meryem Alaoui</option><option value="46">Milena Agus</option><option value="64">Mitch Albom</option><option value="51">Mohammed Aïssaoui</option><option value="36">Nana Kwame Adjei-Brenyah</option><option value="60">Nelly Alard</option><option value="96">Niccolo Ammaniti</option><option value="77">Nina Allan</option><option value="61">Noga Albalach</option><option value="27">Olivier Adam</option><option value="24">Peter Ackroyd</option><option value="41">Pierre Adrian</option><option value="58">Rabih Alameddine</option><option value="70">Raja Alem</option><option value="25">Renata Ada-Ruata</option><option value="39">Ricardo Adolfo</option><option value="73">Robert Alexis</option><option value="56">Ryûnosuke Akutagawa</option><option value="93">Santiago Horacio Amigorena</option><option value="5">Saït Faïk Abasiyanik</option><option value="81">Selva Almada</option><option value="44">Simonetta Agnello Hornby</option><option value="4">Stephan Abarbanell</option><option value="98">Tahmima Anam</option><option value="85">Taleb Alrefai</option><option value="43">Tatamkhulu Afrika</option><option value="52">Tchinguiz Aïtmatov</option><option value="68">Tiit Aleksejev</option><option value="55">Vasili Pavlovitch Aksenov</option><option value="71">Vassilis Alexakis</option><option value="83">Vincent Almendros</option><option value="38">Yassin Adnan</option>
                </select>
            </label>
            <input type="submit" value="Afficher">
        </fieldset>
    </form>
    <!--  Partie Ajouter un Auteur   -->
    <form action="ajouterAuteur.php" method="POST">
        <fieldset>
            <legend>Ajouter un auteur</legend>
            <table style="text-align:right">
                <tr>
                    <td>Nom :</td>
                    <td><input type="text" name="nom" id="nom" value="Onyme"></td>
                </tr>
                <tr>
                    <td>Prénom :</td>
                    <td><input type="text" name="prenom" id="prenom" value=".AAnne"></td>
                </tr>
                <tr>
                    <td>Date de Naissance :</td>
                    <td><input type="text" name="ddn" id="ddn"></td>
                </tr>
                <tr>
                    <td>Date de Décès :</td>
                    <td><input type="text" name="ddd" id="ddd"></td>
                </tr>
                <tr>
                    <td></td><td><input type="submit" value="Ajouter"></td>
                </tr>
            </table>
            
        </fieldset>
    </form>
</body>
</html>