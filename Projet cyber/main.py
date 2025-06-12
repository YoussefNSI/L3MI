"""
Application principale de l'Analyseur de Sites Web.

Ce module initialise et lance l'application d'analyse de sites web.
"""

import sys
import logging
import os
from pathlib import Path

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='analyzer.log',
    filemode='a'
)
logger = logging.getLogger('analyzer.main')

try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTranslator, QLocale
    from PyQt5.QtGui import QIcon
except ImportError as e:
    logger.critical(f"Erreur lors de l'importation des modules PyQt5: {e}")
    print(f"Erreur lors de l'importation des modules PyQt5: {e}")
    sys.exit(1)

# Essayer d'importer le module principal de l'application
try:
    from analyzer import AnalyseWebApp
except ImportError as e:
    logger.critical(f"Erreur lors de l'importation du module analyzer: {e}")
    print(f"Erreur lors de l'importation du module analyzer: {e}")
    sys.exit(1)

# Essayer d'importer le gestionnaire de configuration
try:
    from config_manager import ConfigManager
except ImportError as e:
    logger.critical(f"Erreur lors de l'importation du module config_manager: {e}")
    print(f"Erreur lors de l'importation du module config_manager: {e}")
    sys.exit(1)

def main():
    """Fonction principale de l'application."""
    # Charger la configuration
    config_manager = ConfigManager()
    
    # Créer l'application Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Analyseur de Sites Web")
    app.setOrganizationName("VotreOrganisation")
    
    # Configurer l'icône de l'application si disponible
    icon_path = Path(__file__).parent / "resources" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Configurer la traduction si nécessaire
    translator = QTranslator()
    locale = QLocale.system().name()
    translation_path = Path(__file__).parent / "translations" / f"analyzer_{locale}.qm"
    if translation_path.exists():
        translator.load(str(translation_path))
        app.installTranslator(translator)
    
    # Créer et afficher la fenêtre principale
    window = AnalyseWebApp()
    window.show()
    
    # Exécuter l'application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
