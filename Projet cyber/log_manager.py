"""
Module de gestion des journaux pour l'Analyseur de Sites Web.

Ce module fournit des fonctionnalités pour journaliser les événements de l'application
et afficher ces journaux dans l'interface graphique.
"""

import logging
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCursor

logger = logging.getLogger('analyzer.log')

class LogManager(QObject):
    """
    Gère les journaux de l'application.
    
    Cette classe fournit des méthodes pour journaliser les événements de l'application
    et afficher ces journaux dans l'interface graphique.
    
    Signals:
        log_added (str): Émis lorsqu'un message est ajouté au journal
        log_cleared: Émis lorsque le journal est effacé
        
    Attributes:
        log_widget (QTextEdit): Widget pour afficher les journaux
        log_entries (list): Liste des entrées de journal
        max_entries (int): Nombre maximum d'entrées à conserver
    """
    
    log_added = pyqtSignal(str)
    log_cleared = pyqtSignal()
    
    def __init__(self, log_widget, max_entries=1000):
        """
        Initialise le gestionnaire de journaux.
        
        Args:
            log_widget (QTextEdit): Widget pour afficher les journaux
            max_entries (int, optional): Nombre maximum d'entrées à conserver. Défaut: 1000
        """
        super().__init__()
        self.log_widget = log_widget
        self.log_entries = []
        self.max_entries = max_entries
        logger.info("LogManager initialisé")
        
    def log(self, message, level="INFO"):
        """
        Ajoute un message au journal avec horodatage.
        
        Args:
            message (str): Message à journaliser
            level (str, optional): Niveau de journalisation. Défaut: "INFO"
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        
        # Ajouter à la liste des entrées
        self.log_entries.append(entry)
        
        # Limiter le nombre d'entrées
        if len(self.log_entries) > self.max_entries:
            self.log_entries.pop(0)
        
        # Ajouter au widget
        self.append_to_log(entry)
        
        # Journaliser dans le logger système
        getattr(logger, level.lower(), logger.info)(message)
        
        # Émettre le signal
        self.log_added.emit(entry)
        
    def append_to_log(self, text):
        """
        Ajoute le texte au panneau de log et défile vers le bas.
        
        Args:
            text (str): Texte à ajouter
        """
        try:
            self.log_widget.append(text.rstrip())
            # Force le défilement vers le bas
            scrollbar = self.log_widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout au journal: {str(e)}")
            
    def clear(self):
        """
        Efface le contenu du panneau de log.
        """
        self.log_widget.clear()
        self.log_entries = []
        self.log_cleared.emit()
        logger.info("Journal effacé")
        self.log("Journal effacé", "INFO")
        
    def save_to_file(self, file_path):
        """
        Enregistre le journal dans un fichier.
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            bool: True si l'enregistrement a réussi, False sinon
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_entries))
            logger.info(f"Journal enregistré dans {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du journal: {str(e)}")
            return False
            
    def get_logs_as_text(self):
        """
        Récupère le journal sous forme de texte.
        
        Returns:
            str: Journal sous forme de texte
        """
        return '\n'.join(self.log_entries)
        
    def filter_logs(self, filter_text):
        """
        Filtre les entrées de journal selon un texte.
        
        Args:
            filter_text (str): Texte à rechercher
            
        Returns:
            list: Entrées de journal filtrées
        """
        if not filter_text:
            return self.log_entries
            
        return [entry for entry in self.log_entries if filter_text.lower() in entry.lower()]
