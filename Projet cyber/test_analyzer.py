"""
Tests unitaires pour l'Analyseur de Sites Web.

Ce module contient des tests unitaires pour les différentes classes et fonctionnalités
de l'Analyseur de Sites Web.
"""

import unittest
import os
import tempfile
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Importer les classes à tester
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_manager import ConfigManager
from results_manager import ResultsManager
from log_manager import LogManager

class TestConfigManager(unittest.TestCase):
    """Tests pour la classe ConfigManager."""
    
    def setUp(self):
        """Initialisation des tests."""
        # Créer un fichier temporaire pour la configuration
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.config_manager = ConfigManager(self.temp_file.name)
        
    def tearDown(self):
        """Nettoyage après les tests."""
        # Supprimer le fichier temporaire
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        
    def test_load_default_config(self):
        """Teste le chargement de la configuration par défaut."""
        config = self.config_manager.load()
        self.assertIn('scanner', config)
        self.assertIn('ui', config)
        self.assertIn('reporting', config)
        self.assertIn('logging', config)
        
    def test_set_and_get_config(self):
        """Teste la définition et la récupération de valeurs de configuration."""
        self.config_manager.set('scanner', 'timeout', 20)
        value = self.config_manager.get('scanner', 'timeout')
        self.assertEqual(value, 20)
        
    def test_save_and_load_config(self):
        """Teste l'enregistrement et le chargement de la configuration."""
        self.config_manager.set('scanner', 'new_value', 'test')
        self.config_manager.save()
        
        # Créer une nouvelle instance pour charger la configuration
        new_config_manager = ConfigManager(self.temp_file.name)
        value = new_config_manager.get('scanner', 'new_value')
        self.assertEqual(value, 'test')
        
    def test_reset_to_default(self):
        """Teste la réinitialisation de la configuration aux valeurs par défaut."""
        self.config_manager.set('scanner', 'timeout', 20)
        self.config_manager.reset_to_default()
        value = self.config_manager.get('scanner', 'timeout')
        self.assertEqual(value, 10)  # Valeur par défaut
        

class TestResultsManager(unittest.TestCase):
    """Tests pour la classe ResultsManager."""
    
    def setUp(self):
        """Initialisation des tests."""
        # Créer des mocks pour les widgets
        self.mock_table = MagicMock()
        self.mock_details = MagicMock()
        self.results_manager = ResultsManager(self.mock_table, self.mock_details)
        
    def test_add_result(self):
        """Teste l'ajout d'un résultat."""
        result = {
            'type': 'vulnerability',
            'context': 'DOM_XSS',
            'payload': '<script>alert(1)</script>',
            'url': 'http://example.com'
        }
        
        # Configurer le mock
        self.mock_table.rowCount.return_value = 0
        
        # Ajouter le résultat
        index = self.results_manager.add_result(result)
        
        # Vérifier le résultat
        self.assertEqual(index, 0)
        self.assertEqual(len(self.results_manager.results), 1)
        self.assertEqual(self.results_manager.results[0], result)
        
        # Vérifier que les méthodes du tableau ont été appelées
        self.mock_table.insertRow.assert_called_once_with(0)
        self.mock_table.setItem.assert_called()
        
    def test_clear_results(self):
        """Teste l'effacement des résultats."""
        # Ajouter un résultat
        result = {'type': 'vulnerability', 'context': 'DOM_XSS'}
        self.results_manager.results = [result]
        
        # Effacer les résultats
        self.results_manager.clear_results()
        
        # Vérifier le résultat
        self.assertEqual(len(self.results_manager.results), 0)
        self.mock_table.setRowCount.assert_called_once_with(0)
        self.mock_details.clear.assert_called_once()
        
    def test_show_result_details(self):
        """Teste l'affichage des détails d'un résultat."""
        # Ajouter un résultat
        result = {
            'type': 'vulnerability',
            'context': 'DOM_XSS',
            'payload': '<script>alert(1)</script>',
            'url': 'http://example.com'
        }
        self.results_manager.results = [result]
        
        # Afficher les détails
        self.results_manager.show_result_details(0)
        
        # Vérifier que la méthode setMarkdown a été appelée
        self.mock_details.setMarkdown.assert_called_once()
        
    def test_generate_summary(self):
        """Teste la génération d'un résumé."""
        # Ajouter des résultats
        results = [
            {'type': 'vulnerability', 'context': 'DOM_XSS'},
            {'type': 'info', 'context': 'INFO'},
            {'type': 'vulnerability', 'context': 'REFLECTED'}
        ]
        self.results_manager.results = results
        
        # Générer le résumé
        summary = self.results_manager.generate_summary()
        
        # Vérifier le résumé
        self.assertIn("Résultats trouvés: 3", summary)
        self.assertIn("Vulnérabilités détectées: 2", summary)
        

class TestLogManager(unittest.TestCase):
    """Tests pour la classe LogManager."""
    
    def setUp(self):
        """Initialisation des tests."""
        # Créer un mock pour le widget
        self.mock_log_widget = MagicMock()
        self.log_manager = LogManager(self.mock_log_widget)
        
    def test_log(self):
        """Teste l'ajout d'un message au journal."""
        # Ajouter un message
        self.log_manager.log("Test message")
        
        # Vérifier le résultat
        self.assertEqual(len(self.log_manager.log_entries), 1)
        self.assertIn("Test message", self.log_manager.log_entries[0])
        self.mock_log_widget.append.assert_called_once()
        
    def test_clear(self):
        """Teste l'effacement du journal."""
        # Ajouter un message
        self.log_manager.log_entries = ["Test"]
        
        # Effacer le journal
        self.log_manager.clear()
        
        # Vérifier le résultat
        self.assertEqual(len(self.log_manager.log_entries), 1)  # Un message "Journal effacé" est ajouté
        self.mock_log_widget.clear.assert_called_once()
        
    def test_filter_logs(self):
        """Teste le filtrage des entrées de journal."""
        # Ajouter des messages
        self.log_manager.log_entries = [
            "[12:00:00] [INFO] Message d'information",
            "[12:01:00] [ERROR] Message d'erreur",
            "[12:02:00] [INFO] Autre message"
        ]
        
        # Filtrer les messages
        filtered = self.log_manager.filter_logs("erreur")
        
        # Vérifier le résultat
        self.assertEqual(len(filtered), 1)
        self.assertIn("erreur", filtered[0])
        

if __name__ == '__main__':
    unittest.main()
