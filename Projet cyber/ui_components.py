"""
Module de gestion des composants UI pour l'Analyseur de Sites Web.

Ce module fournit des classes pour gérer les différents composants de
l'interface utilisateur de l'application.
"""

import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                            QPushButton, QRadioButton, QButtonGroup, QFileDialog,
                            QFrame, QProgressBar, QCheckBox, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QIcon, QFont

logger = logging.getLogger('analyzer.ui')

class SourceSelectorWidget(QGroupBox):
    """
    Widget pour sélectionner la source de données (URL ou fichier).
    
    Signals:
        source_changed (str): Émis lorsque la source est modifiée ('url' ou 'file')
        url_changed (str): Émis lorsque l'URL est modifiée
        file_changed (str): Émis lorsque le fichier est modifié
        
    Attributes:
        url_radio (QRadioButton): Bouton radio pour sélectionner l'URL
        file_radio (QRadioButton): Bouton radio pour sélectionner le fichier
        url_input (QLineEdit): Champ de saisie de l'URL
        file_input (QLineEdit): Champ de saisie du fichier
        browse_button (QPushButton): Bouton pour parcourir les fichiers
    """
    
    source_changed = pyqtSignal(str)  # 'url' ou 'file'
    url_changed = pyqtSignal(str)
    file_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Initialise le widget de sélection de source.
        
        Args:
            parent: Widget parent
        """
        super().__init__("Source de données", parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Configure l'interface utilisateur du widget."""
        layout = QVBoxLayout(self)
        
        # Options de source
        source_options_layout = QHBoxLayout()
        self.url_radio = QRadioButton("URL du site web")
        self.file_radio = QRadioButton("Fichier HTML local")
        self.url_radio.setChecked(True)
        
        group = QButtonGroup(self)
        group.addButton(self.url_radio)
        group.addButton(self.file_radio)
        
        source_options_layout.addWidget(self.url_radio)
        source_options_layout.addWidget(self.file_radio)
        source_options_layout.addStretch()
        layout.addLayout(source_options_layout)
        
        # Entrée URL
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://exemple.com")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input, 1)
        layout.addLayout(url_layout)
        
        # Entrée fichier
        file_layout = QHBoxLayout()
        file_label = QLabel("Fichier:")
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Sélectionnez un fichier HTML")
        self.file_input.setReadOnly(True)
        self.browse_button = QPushButton("Parcourir...")
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_input, 1)
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)
        
        # Connexion des signaux
        self.url_radio.toggled.connect(self._on_source_toggled)
        self.file_radio.toggled.connect(self._on_source_toggled)
        self.url_input.textChanged.connect(self.url_changed)
        self.file_input.textChanged.connect(self.file_changed)
        self.browse_button.clicked.connect(self._browse_file)
        
        # État initial
        self._on_source_toggled()
        
    @pyqtSlot()
    def _on_source_toggled(self):
        """Gère le changement de source."""
        url_enabled = self.url_radio.isChecked()
        self.url_input.setEnabled(url_enabled)
        self.file_input.setEnabled(not url_enabled)
        self.browse_button.setEnabled(not url_enabled)
        
        source_type = "url" if url_enabled else "file"
        self.source_changed.emit(source_type)
        
    @pyqtSlot()
    def _browse_file(self):
        """Ouvre une boîte de dialogue pour sélectionner un fichier HTML."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Sélectionner un fichier HTML", "", "Fichiers HTML (*.html *.htm *.php)"
            )
            if file_path:
                self.file_input.setText(file_path)
        except Exception as e:
            logger.error(f"Erreur lors de la sélection du fichier: {str(e)}")
            QMessageBox.warning(self, "Erreur", f"Impossible de sélectionner le fichier: {str(e)}")
            
    def get_source_type(self):
        """
        Récupère le type de source sélectionné.
        
        Returns:
            str: 'url' ou 'file'
        """
        return "url" if self.url_radio.isChecked() else "file"
        
    def get_source_value(self):
        """
        Récupère la valeur de la source sélectionnée.
        
        Returns:
            str: URL ou chemin du fichier
        """
        if self.url_radio.isChecked():
            return self.url_input.text().strip()
        else:
            return self.file_input.text().strip()
            
    def set_enabled(self, enabled):
        """
        Active ou désactive le widget.
        
        Args:
            enabled (bool): True pour activer, False pour désactiver
        """
        self.url_radio.setEnabled(enabled)
        self.file_radio.setEnabled(enabled)
        self.url_input.setEnabled(enabled and self.url_radio.isChecked())
        self.file_input.setEnabled(enabled and self.file_radio.isChecked())
        self.browse_button.setEnabled(enabled and self.file_radio.isChecked())


class AnalysisControlWidget(QGroupBox):
    """
    Widget pour contrôler l'analyse (démarrage, arrêt, progression).
    
    Signals:
        static_analysis_requested: Émis lorsque l'analyse statique est demandée
        dynamic_analysis_requested: Émis lorsque l'analyse dynamique est demandée
        stop_requested: Émis lorsque l'arrêt de l'analyse est demandé
        headless_toggled (bool): Émis lorsque l'option headless est modifiée
        
    Attributes:
        static_button (QPushButton): Bouton pour lancer l'analyse statique
        dynamic_button (QPushButton): Bouton pour lancer l'analyse dynamique
        stop_button (QPushButton): Bouton pour arrêter l'analyse
        progress_bar (QProgressBar): Barre de progression de l'analyse
        headless_checkbox (QCheckBox): Case à cocher pour le mode headless
    """
    
    static_analysis_requested = pyqtSignal()
    dynamic_analysis_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    headless_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        """
        Initialise le widget de contrôle d'analyse.
        
        Args:
            parent: Widget parent
        """
        super().__init__("Type d'analyse", parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Configure l'interface utilisateur du widget."""
        layout = QVBoxLayout(self)
        
        # Boutons d'analyse
        buttons_layout = QHBoxLayout()
        self.static_button = QPushButton("Analyse Statique")
        self.dynamic_button = QPushButton("Analyse Dynamique")
        self.stop_button = QPushButton("Arrêter l'analyse")
        self.stop_button.setEnabled(False)
        
        buttons_layout.addWidget(self.static_button)
        buttons_layout.addWidget(self.dynamic_button)
        buttons_layout.addWidget(self.stop_button)
        layout.addLayout(buttons_layout)
        
        # Barre de progression
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progression:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar, 1)
        layout.addLayout(progress_layout)
        
        # Option headless
        headless_layout = QHBoxLayout()
        self.headless_checkbox = QCheckBox("Mode headless (navigateur invisible)")
        self.headless_checkbox.setChecked(True)
        headless_layout.addWidget(self.headless_checkbox)
        layout.addLayout(headless_layout)
        
        # Connexion des signaux
        self.static_button.clicked.connect(self.static_analysis_requested)
        self.dynamic_button.clicked.connect(self.dynamic_analysis_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.headless_checkbox.toggled.connect(self.headless_toggled)
        
    def set_progress(self, value):
        """
        Définit la valeur de la barre de progression.
        
        Args:
            value (int): Valeur de progression (0-100)
        """
        self.progress_bar.setValue(value)
        
    def set_analyzing_state(self, analyzing):
        """
        Met à jour l'état du widget selon que l'analyse est en cours ou non.
        
        Args:
            analyzing (bool): True si une analyse est en cours, False sinon
        """
        self.static_button.setEnabled(not analyzing)
        self.dynamic_button.setEnabled(not analyzing)
        self.stop_button.setEnabled(analyzing)
        self.headless_checkbox.setEnabled(not analyzing)
        
    def is_headless(self):
        """
        Indique si le mode headless est activé.
        
        Returns:
            bool: True si le mode headless est activé, False sinon
        """
        return self.headless_checkbox.isChecked()
