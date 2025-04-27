import sys
import io
import re
import os
import threading
import traceback
import time
import urllib.parse
from datetime import datetime
from queue import Queue, Empty

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QLineEdit, QPushButton,
                                 QRadioButton, QButtonGroup, QFileDialog, QFrame,
                                 QTextEdit, QSplitter, QProgressBar, QTableWidget,
                                 QTableWidgetItem, QHeaderView, QMessageBox,
                                 QCheckBox, QStatusBar, QTabWidget)
    from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QObject, QThread, QTimer
    from PyQt5.QtGui import QTextCursor, QColor, QIcon
except ImportError as e:
    print(f"Erreur lors de l'importation des modules PyQt5: {e}")
    sys.exit(1)

# Importation sécurisée du scanner XSS
try:
    from v2 import AdvancedXSSScanner
except ImportError as e:
    print(f"Erreur lors de l'importation du module AdvancedXSSScanner: {e}")
    print("Assurez-vous que le fichier v2.py est dans le même répertoire.")
    sys.exit(1)

# Constantes pour les messages d'erreur et textes d'interface
MESSAGES = {
    "welcome": "Bienvenue dans l'Analyseur de Sites Web",
    "ready": "Sélectionnez une source et lancez une analyse",
    "analysis_running": "Une analyse est déjà en cours. Veuillez attendre qu'elle se termine ou l'arrêter.",
    "analysis_stopped": "Analyse arrêtée",
    "analysis_complete": "Analyse terminée: {} résultats",
    "analysis_error": "Erreur lors de l'analyse",
    "results_cleared": "Les résultats ont été effacés.",
    "results_saved": "Les résultats ont été enregistrés dans {}",
}

# Palette de couleurs pour les résultats
COLORS = {
    "vulnerability": QColor(255, 200, 200),  # Rouge clair
    "potential": QColor(255, 255, 200),      # Jaune clair
    "info": QColor(200, 255, 200),           # Vert clair
}


# Classe pour rediriger la sortie standard vers le widget de log
class OutputRedirector(QObject):
    """Redirige la sortie standard vers un signal Qt."""
    output_written = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.buffer = io.StringIO()

    def write(self, text):
        if text.strip():
            self.output_written.emit(text)
        self.buffer.write(text)

    def flush(self):
        pass


# Classe de travailleur pour exécuter l'analyse en arrière-plan
class AnalysisWorker(QThread):
    """Classe de travailleur pour exécuter l'analyse en arrière-plan."""
    progress_updated = pyqtSignal(int)
    result_found = pyqtSignal(dict)
    analysis_completed = pyqtSignal(list)
    analysis_error = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, analysis_type, source_type, url_or_file, headless=True):
        """
        Initialise le travailleur d'analyse.
        
        :param analysis_type: Type d'analyse ('static' ou 'dynamic')
        :param source_type: Type de source ('url' ou 'file') 
        :param url_or_file: URL ou chemin du fichier à analyser
        :param headless: Si True, lance le navigateur en mode headless
        """
        super().__init__()
        self.analysis_type = analysis_type
        self.source_type = source_type
        self.url_or_file = url_or_file
        self.headless = headless
        self.results = []
        self.terminate_flag = False

    def run(self):
        """Exécute l'analyse dans un thread séparé."""
        try:
            self.log_message.emit(f"Démarrage de l'analyse {self.analysis_type}...")

            # Méthode d'analyse selon le type de source
            if self.source_type == 'url':
                self.analyze_url()
            elif self.source_type == 'file':
                self.analyze_file()

            # Signaler la fin de l'analyse avec tous les résultats
            if not self.terminate_flag:
                self.analysis_completed.emit(self.results)
                self.log_message.emit("Analyse terminée avec succès.")
            else:
                self.log_message.emit("Analyse interrompue par l'utilisateur.")

        except Exception as e:
            error_msg = f"Erreur lors de l'analyse: {str(e)}\n{traceback.format_exc()}"
            self.analysis_error.emit(error_msg)
            self.log_message.emit(error_msg)

    def analyze_url(self):
        """Analyse une URL avec le scanner XSS."""
        try:
            # Création et configuration du scanner
            scanner = AdvancedXSSScanner(self.url_or_file, self.headless)

            # Hook pour recevoir les résultats et mettre à jour la progression
            original_log_vulnerability = scanner.log_vulnerability

            def custom_log_vulnerability(context, payload, url=None):
                result = original_log_vulnerability(context, payload, url)
                self.result_found.emit({
                    'type': 'vulnerability',
                    'context': context,
                    'payload': payload,
                    'url': url or self.url_or_file
                })
                self.results.append(result)
                return result

            # Remplacer la méthode originale par notre méthode personnalisée
            scanner.log_vulnerability = custom_log_vulnerability

            # Exécuter le scan approprié
            if self.analysis_type == 'static':
                scanner.scan()
            else:  # dynamic
                scanner.test_dynamic_analysis(self.url_or_file)

            # Générer et renvoyer le rapport
            return scanner.generate_report()

        except Exception as e:
            raise Exception(f"Erreur lors de l'analyse de l'URL: {str(e)}")

    def analyze_file(self):
        """Analyse un fichier HTML local."""
        try:
            # Lire le contenu du fichier HTML
            with open(self.url_or_file, 'r', encoding='utf-8') as file:
                html_content = file.read()

            # Pour l'analyse de fichier local, on peut:
            # 1. Créer un serveur web local temporaire et y servir le fichier
            # 2. Analyser directement le contenu HTML

            # Méthode simplifiée: extraction et analyse des éléments à risque
            # Dans une application réelle, vous voudriez implémenter un serveur web temporaire

            vulnerabilities = []

            # Vérifier les scripts potentiellement dangereux
            script_tags = re.findall(r'<script.*?>(.*?)</script>', html_content, re.DOTALL)
            total_items = len(script_tags)
            
            for i, script in enumerate(script_tags):
                if 'eval(' in script or 'document.cookie' in script or 'localStorage' in script:
                    result = {
                        'type': 'vulnerability',
                        'context': 'script',
                        'payload': script[:50] + ('...' if len(script) > 50 else ''),
                        'url': f"{self.url_or_file}#script-{i}"
                    }
                    self.result_found.emit(result)
                    vulnerabilities.append(result)

                # Simuler la progression
                self.progress_updated.emit(int((i + 1) / max(total_items, 1) * 50))

                # Vérifier si l'utilisateur a demandé l'arrêt
                if self.terminate_flag:
                    return

            # Vérifier les entrées de formulaire
            input_tags = re.findall(r'<input.*?>', html_content)
            for i, input_tag in enumerate(input_tags):
                if 'type="text"' in input_tag or 'type="search"' in input_tag:
                    result = {
                        'type': 'potential',
                        'context': 'input',
                        'payload': input_tag[:50] + ('...' if len(input_tag) > 50 else ''),
                        'url': f"{self.url_or_file}#input-{i}"
                    }
                    self.result_found.emit(result)
                    vulnerabilities.append(result)
                
                # Mettre à jour la progression (50-100%)
                progress = 50 + int((i + 1) / max(len(input_tags), 1) * 50)
                self.progress_updated.emit(progress)
                
                if self.terminate_flag:
                    return

            return vulnerabilities

        except Exception as e:
            raise Exception(f"Erreur lors de l'analyse du fichier: {str(e)}")

    def terminate(self):
        """Arrête l'analyse en cours de façon propre."""
        self.terminate_flag = True
        super().terminate()


class AnalyseWebApp(QMainWindow):
    """Application d'analyse de sites web pour détecter les vulnérabilités XSS."""
    
    def __init__(self):
        """Initialise l'interface utilisateur de l'application."""
        super().__init__()
        self.setWindowTitle("Analyseur de Sites Web")
        self.setMinimumSize(900, 600)

        # Créer la barre de statut
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # État actuel de l'analyse
        self.analysis_running = False
        self.current_worker = None
        self.results_data = []

        # Widget central avec splitter
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)

        # Panneau gauche
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(20, 20, 20, 20)

        # Section source de données
        source_group = QFrame()
        source_group.setFrameShape(QFrame.StyledPanel)
        source_layout = QVBoxLayout(source_group)

        source_title = QLabel("Source de données")
        source_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        source_layout.addWidget(source_title)

        source_options_layout = QHBoxLayout()
        self.url_radio = QRadioButton("URL du site web")
        self.file_radio = QRadioButton("Fichier HTML local")
        self.url_radio.setChecked(True)

        source_group_btn = QButtonGroup(self)
        source_group_btn.addButton(self.url_radio)
        source_group_btn.addButton(self.file_radio)

        source_options_layout.addWidget(self.url_radio)
        source_options_layout.addWidget(self.file_radio)
        source_options_layout.addStretch()
        source_layout.addLayout(source_options_layout)

        # Entrée URL
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://exemple.com")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input, 1)
        source_layout.addLayout(url_layout)

        # Entrée fichier
        file_layout = QHBoxLayout()
        file_label = QLabel("Fichier:")
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Sélectionnez un fichier HTML")
        self.file_input.setReadOnly(True)
        browse_button = QPushButton("Parcourir...")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_input, 1)
        file_layout.addWidget(browse_button)
        source_layout.addLayout(file_layout)

        # Option pour mode headless (navigateur invisible)
        headless_layout = QHBoxLayout()
        self.headless_checkbox = QCheckBox("Mode headless (navigateur invisible)")
        self.headless_checkbox.setChecked(True)
        headless_layout.addWidget(self.headless_checkbox)
        source_layout.addLayout(headless_layout)

        left_layout.addWidget(source_group)

        # Section type d'analyse
        analysis_group = QFrame()
        analysis_group.setFrameShape(QFrame.StyledPanel)
        analysis_layout = QVBoxLayout(analysis_group)

        analysis_title = QLabel("Type d'analyse")
        analysis_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        analysis_layout.addWidget(analysis_title)

        buttons_layout = QHBoxLayout()
        self.static_button = QPushButton("Analyse Statique")
        self.dynamic_button = QPushButton("Analyse Dynamique")
        self.stop_button = QPushButton("Arrêter l'analyse")
        self.stop_button.setEnabled(False)

        self.static_button.clicked.connect(self.run_static_analysis)
        self.dynamic_button.clicked.connect(self.run_dynamic_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)

        buttons_layout.addWidget(self.static_button)
        buttons_layout.addWidget(self.dynamic_button)
        buttons_layout.addWidget(self.stop_button)
        analysis_layout.addLayout(buttons_layout)

        # Barre de progression
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progression:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar, 1)
        analysis_layout.addLayout(progress_layout)

        left_layout.addWidget(analysis_group)

        # Section résultats
        results_group = QFrame()
        results_group.setFrameShape(QFrame.StyledPanel)
        results_layout = QVBoxLayout(results_group)

        results_title = QLabel("Résultats")
        results_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        results_layout.addWidget(results_title)

        # Tableau de résultats
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Type", "Contexte", "URL/Charge utile"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.itemClicked.connect(self.show_result_details)
        results_layout.addWidget(self.results_table)

        # Boutons pour les actions sur les résultats
        results_buttons_layout = QHBoxLayout()
        self.clear_results_button = QPushButton("Effacer les résultats")
        self.save_results_button = QPushButton("Enregistrer les résultats")

        self.clear_results_button.clicked.connect(self.clear_results)
        self.save_results_button.clicked.connect(self.save_results)

        results_buttons_layout.addWidget(self.clear_results_button)
        results_buttons_layout.addWidget(self.save_results_button)
        results_layout.addLayout(results_buttons_layout)

        left_layout.addWidget(results_group, 1)

        # Connecter les signaux
        self.url_radio.toggled.connect(self.toggle_source_inputs)
        self.file_radio.toggled.connect(self.toggle_source_inputs)

        main_splitter.addWidget(left_widget)

        # Panneau droit avec onglets
        right_widget = QTabWidget()

        # Onglet log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)

        log_title = QLabel("Déroulement du scanner")
        log_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        log_layout.addWidget(log_title)

        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setStyleSheet("font-family: monospace; background-color: #f5f5f5;")
        self.log_panel.setPlaceholderText("Le déroulement du scanner s'affichera ici...")
        log_layout.addWidget(self.log_panel)

        # Bouton pour effacer le log
        clear_log_button = QPushButton("Effacer le journal")
        clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_button)

        right_widget.addTab(log_widget, "Journal")

        # Onglet détails
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("font-family: monospace;")
        details_layout.addWidget(self.details_text)

        right_widget.addTab(details_widget, "Détails")

        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])

        # Initialiser les états de l'interface
        self.toggle_source_inputs()
        self.setup_output_redirection()

        # Timer pour les mises à jour régulières de l'interface
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(500)  # Mise à jour toutes les 500ms

        # Afficher un message d'accueil
        self.status_bar.showMessage("Prêt à analyser", 3000)
        self.log(MESSAGES["welcome"])
        self.log(MESSAGES["ready"])

    def setup_output_redirection(self):
        """Configure la redirection de la sortie standard vers le panneau de log de manière sécurisée"""
        try:
            self.stdout_redirector = OutputRedirector()
            self.stdout_redirector.output_written.connect(self.append_to_log)
            # Sauvegarder la référence originale à stdout
            self.original_stdout = sys.stdout
            # Rediriger stdout vers notre redirecteur
            sys.stdout = self.stdout_redirector
        except Exception as e:
            print(f"Erreur lors de la configuration de la redirection de sortie: {str(e)}")

    def append_to_log(self, text):
        """Ajoute le texte au panneau de log et défile vers le bas"""
        try:
            self.log_panel.append(text.rstrip())
            # Force le défilement vers le bas
            scrollbar = self.log_panel.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            # En cas d'erreur, utilise la sortie standard originale
            original_stdout = getattr(self, 'original_stdout', sys.stdout)
            original_stdout.write(f"Erreur lors de l'ajout au journal: {str(e)}\n")

    def log(self, message):
        """Ajoute un message au journal avec horodatage"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.append_to_log(f"[{timestamp}] {message}")

    def clear_log(self):
        """Efface le contenu du panneau de log"""
        self.log_panel.clear()
        self.log("Journal effacé")

    @pyqtSlot()
    def toggle_source_inputs(self):
        """Active/désactive les champs selon la source sélectionnée"""
        url_enabled = self.url_radio.isChecked()
        self.url_input.setEnabled(url_enabled)
        self.file_input.setEnabled(not url_enabled)

    @pyqtSlot()
    def browse_file(self):
        """Ouvre une boîte de dialogue pour sélectionner un fichier HTML"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Sélectionner un fichier HTML", "", "Fichiers HTML (*.html *.htm)"
            )
            if file_path:
                self.file_input.setText(file_path)
        except Exception as e:
            self.log(f"Erreur lors de la sélection du fichier: {str(e)}")
            QMessageBox.warning(self, "Erreur", f"Impossible de sélectionner le fichier: {str(e)}")

    def validate_url(self, url):
        """
        Valide le format de l'URL
        
        :param url: URL à valider
        :return: tuple (is_valid, message)
        """
        if not url:
            return False, "L'URL ne peut pas être vide"

        # Vérifier si l'URL commence par http:// ou https://
        if not url.startswith(('http://', 'https://')):
            return False, "L'URL doit commencer par http:// ou https://"

        try:
            # Analyser l'URL pour vérifier sa structure
            parsed_url = urllib.parse.urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                return False, "URL invalide: schéma ou domaine manquant"
                
            # Tester les caractères spéciaux dans le nom de domaine
            if not re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", parsed_url.netloc.split(':')[0]):
                return False, "Le nom de domaine contient des caractères invalides"

            return True, "URL valide"
        except Exception as e:
            return False, f"URL invalide: {str(e)}"

    def validate_file(self, file_path):
        """
        Valide le chemin du fichier
        
        :param file_path: Chemin du fichier à valider
        :return: tuple (is_valid, message)
        """
        if not file_path:
            return False, "Le chemin du fichier ne peut pas être vide"

        if not os.path.exists(file_path):
            return False, "Le fichier n'existe pas"

        if not os.path.isfile(file_path):
            return False, "Le chemin spécifié n'est pas un fichier"

        if not file_path.lower().endswith(('.html', '.htm')):
            return False, "Le fichier doit être au format HTML (.html ou .htm)"
            
        # Vérifier si le fichier est lisible
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1)
            return True, "Fichier valide"
        except Exception as e:
            return False, f"Le fichier n'est pas lisible: {str(e)}"

    @pyqtSlot()
    def run_static_analysis(self):
        """Exécute l'analyse statique"""
        self.run_analysis('static')

        # à implémenter

    @pyqtSlot()
    def run_dynamic_analysis(self):
        """Exécute l'analyse dynamique"""
        self.run_analysis('dynamic')

    def run_analysis(self, analysis_type):
        """
        Méthode commune pour exécuter une analyse
        
        :param analysis_type: Type d'analyse ('static' ou 'dynamic')
        """
        # Vérifier si une analyse est déjà en cours
        if self.analysis_running:
            QMessageBox.warning(self, "Analyse en cours", MESSAGES["analysis_running"])
            return

        # Déterminer la source et valider les entrées
        if self.url_radio.isChecked():
            url = self.url_input.text().strip()
            valid, message = self.validate_url(url)
            if not valid:
                self.log(f"Erreur: {message}")
                QMessageBox.warning(self, "URL invalide", message)
                return
            source_type = 'url'
            source_value = url
        else:
            file_path = self.file_input.text()
            valid, message = self.validate_file(file_path)
            if not valid:
                self.log(f"Erreur: {message}")
                QMessageBox.warning(self, "Fichier invalide", message)
                return
            source_type = 'file'
            source_value = file_path

        # Réinitialiser l'interface pour une nouvelle analyse
        self.clear_log()
        self.progress_bar.setValue(0)

        # Mettre à jour l'état de l'interface
        self.set_ui_analyzing_state(True)

        # Message de début d'analyse
        analysis_name = "statique" if analysis_type == 'static' else "dynamique"
        self.log(f"Démarrage de l'analyse {analysis_name}...")
        self.log(f"Source: {'URL' if source_type == 'url' else 'Fichier'} - {source_value}")

        # Créer et démarrer le worker
        headless = self.headless_checkbox.isChecked()
        self.current_worker = AnalysisWorker(analysis_type, source_type, source_value, headless)

        # Connecter les signaux
        self.current_worker.progress_updated.connect(self.update_progress)
        self.current_worker.result_found.connect(self.add_result)
        self.current_worker.analysis_completed.connect(self.on_analysis_completed)
        self.current_worker.analysis_error.connect(self.on_analysis_error)
        self.current_worker.log_message.connect(self.log)

        # Démarrer l'analyse en arrière-plan
        self.current_worker.start()

    @pyqtSlot()
    def stop_analysis(self):
        """Arrête l'analyse en cours"""
        if self.current_worker and self.analysis_running:
            self.log("Arrêt de l'analyse demandé...")
            self.current_worker.terminate()
            self.set_ui_analyzing_state(False)
            self.status_bar.showMessage(MESSAGES["analysis_stopped"], 3000)

    def set_ui_analyzing_state(self, analyzing):
        """
        Met à jour l'état de l'interface selon que l'analyse est en cours ou non
        
        :param analyzing: True si une analyse est en cours, False sinon
        """
        self.analysis_running = analyzing

        # Désactiver/activer les contrôles d'entrée pendant l'analyse
        self.url_radio.setEnabled(not analyzing)
        self.file_radio.setEnabled(not analyzing)
        self.url_input.setEnabled(not analyzing and self.url_radio.isChecked())
        self.file_input.setEnabled(not analyzing and self.file_radio.isChecked())
        self.static_button.setEnabled(not analyzing)
        self.dynamic_button.setEnabled(not analyzing)
        self.headless_checkbox.setEnabled(not analyzing)

        # Activer/désactiver le bouton d'arrêt
        self.stop_button.setEnabled(analyzing)

        if analyzing:
            self.status_bar.showMessage("Analyse en cours...")
        else:
            self.status_bar.showMessage("Prêt", 3000)

    @pyqtSlot(int)
    def update_progress(self, value):
        """Met à jour la barre de progression"""
        self.progress_bar.setValue(value)

    @pyqtSlot(dict)
    def add_result(self, result_data):
        """
        Ajoute un résultat au tableau des résultats
        
        :param result_data: Dictionnaire contenant les informations du résultat
        """
        try:
            # Ajouter aux données stockées
            self.results_data.append(result_data)

            # Ajouter une nouvelle ligne au tableau
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            # Remplir les cellules
            self.results_table.setItem(row, 0, QTableWidgetItem(result_data.get('type', 'Inconnu')))
            self.results_table.setItem(row, 1, QTableWidgetItem(result_data.get('context', 'Inconnu')))

            # Combiner URL et payload pour la troisième colonne
            payload_info = result_data.get('url', '')
            if result_data.get('payload'):
                payload_preview = result_data['payload'][:30]
                if len(result_data['payload']) > 30:
                    payload_preview += "..."
                payload_info += f" - {payload_preview}"

            self.results_table.setItem(row, 2, QTableWidgetItem(payload_info))

            # Colorer la ligne en fonction du type de résultat
            result_type = result_data.get('type', 'info')
            color = COLORS.get(result_type, QColor(255, 255, 255))
            
            for col in range(3):
                self.results_table.item(row, col).setBackground(color)

            # Mettre en surbrillance la ligne
            self.results_table.selectRow(row)

            # Défilement vers le bas du tableau
            self.results_table.scrollToBottom()

            # Mise à jour du log
            self.log(f"Trouvé: {result_data.get('context')} - {result_data.get('payload', '')[:30]}")

        except Exception as e:
            self.log(f"Erreur lors de l'ajout d'un résultat: {str(e)}")

    def show_result_details(self, item):
        """
        Affiche les détails d'un résultat sélectionné dans l'onglet Détails
        
        :param item: Élément de tableau cliqué
        """
        row = item.row()
        if row < 0 or row >= len(self.results_data):
            return
            
        result = self.results_data[row]
        details = (f"# Détails du résultat\n\n"
                   f"- **Type:** {result.get('type', 'Inconnu')}\n"
                   f"- **Contexte:** {result.get('context', 'Inconnu')}\n"
                   f"- **URL:** {result.get('url', 'N/A')}\n\n"
                   f"## Charge utile\n\n```\n{result.get('payload', 'N/A')}\n"
                   f"```")
        self.details_text.setMarkdown(details)

    @pyqtSlot(list)
    def on_analysis_completed(self, results):
        """Traite la fin de l'analyse"""
        self.set_ui_analyzing_state(False)
        self.progress_bar.setValue(100)

        count = len(results)
        vuln_count = sum(1 for r in self.results_data if r.get('type') == 'vulnerability')

        self.log(f"Analyse terminée. Trouvé {count} résultats, dont {vuln_count} vulnérabilités.")
        self.status_bar.showMessage(MESSAGES["analysis_complete"].format(count), 5000)

        # Mise à jour des détails
        summary = (f"# Résumé de l'analyse\n\n"
                   f"- **Date et heure:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"- **Type d'analyse:** {'Statique' if self.current_worker.analysis_type == 'static' else 'Dynamique'}\n"
                   f"- **Source:** {'URL' if self.current_worker.source_type == 'url' else 'Fichier local'}\n"
                   f"- **Cible:** {self.current_worker.url_or_file}\n"
                   f"- **Résultats trouvés:** {count}\n"
                   f"- **Vulnérabilités détectées:** {vuln_count}\n\n"
                   f"## Détails des résultats\n\n")

        for i, result in enumerate(self.results_data):
            summary += (f"### Résultat #{i + 1}\n"
                        f"- **Type:** {result.get('type', 'Inconnu')}\n"
                        f"- **Contexte:** {result.get('context', 'Inconnu')}\n"
                        f"- **URL:** {result.get('url', 'N/A')}\n"
                        f"- **Payload:** {result.get('payload', 'N/A')}\n\n")

        self.details_text.setMarkdown(summary)

    @pyqtSlot(str)
    def on_analysis_error(self, error_message):
        """Traite les erreurs d'analyse"""
        self.set_ui_analyzing_state(False)
        self.log(f"ERREUR: {error_message}")
        self.status_bar.showMessage(MESSAGES["analysis_error"], 5000)
        QMessageBox.critical(self, "Erreur d'analyse",
                             f"Une erreur est survenue pendant l'analyse:\n\n{error_message}")

    def clear_results(self):
        """
        Efface tous les résultats actuels de la table et réinitialise les données stockées.
        """
        self.results_data = []
        self.results_table.setRowCount(0)
        self.details_text.clear()
        self.log(MESSAGES["results_cleared"])

    def save_results(self):
        """Enregistre les résultats dans un fichier"""
        try:
            if not self.results_data:
                raise Exception("Aucun résultat à enregistrer.")

            file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer les résultats", "", "Fichiers texte (*.txt);;Tous les fichiers (*)")
            if not file_path:
                return  # L'utilisateur a annulé l'enregistrement

            with open(file_path, 'w', encoding='utf-8') as file:
                for result in self.results_data:
                    file.write(f"Type: {result.get('type', 'Inconnu')}\n")
                    file.write(f"Contexte: {result.get('context', 'Inconnu')}\n")
                    file.write(f"URL: {result.get('url', 'N/A')}\n")
                    file.write(f"Payload: {result.get('payload', 'N/A')}\n")
                    file.write("\n")  # Ligne vide entre les résultats

            self.log(MESSAGES["results_saved"].format(file_path))

        except Exception as e:
            self.log(f"Erreur lors de l'enregistrement des résultats: {str(e)}")
            QMessageBox.warning(self, "Erreur d'enregistrement", f"Impossible d'enregistrer les résultats: {str(e)}")

    def update_ui(self):
        """
        Met à jour périodiquement l'interface utilisateur.
        Cette méthode est appelée régulièrement par le QTimer.
        """
        # Mettre à jour le chronomètre si l'analyse est en cours
        if self.analysis_running:
            # Vous pouvez ajouter ici du code pour mettre à jour un chronomètre
            # ou toute autre information dynamique de l'interface
            pass

        # Vérifier si le worker est toujours en cours d'exécution
        if self.current_worker and not self.current_worker.isRunning() and self.analysis_running:
            # L'analyse s'est terminée ou a été arrêtée sans notification
            self.set_ui_analyzing_state(False)
            self.status_bar.showMessage("Analyse terminée ou interrompue", 3000)
            # Réinitialiser le worker
            self.current_worker = None


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = AnalyseWebApp()
    window.show()
    sys.exit(app.exec_())
