"""
Module de gestion de la configuration pour l'Analyseur de Sites Web.

Ce module fournit des fonctionnalités pour charger, enregistrer et manipuler
la configuration de l'application.
"""

import json
import os
import logging
from pathlib import Path

logger = logging.getLogger('analyzer.config')

class ConfigManager:
    """
    Gère la configuration de l'application.
    
    Cette classe fournit des méthodes pour charger, enregistrer et manipuler
    la configuration de l'application à partir d'un fichier JSON.
    
    Attributes:
        config_file (str): Chemin vers le fichier de configuration
        config (dict): Configuration actuelle
        default_config (dict): Configuration par défaut
    """
    
    def __init__(self, config_file="config.json"):
        """
        Initialise le gestionnaire de configuration.
        
        Args:
            config_file (str, optional): Chemin vers le fichier de configuration. Défaut: "config.json"
        """
        self.config_file = config_file
        self.default_config = {
            "scanner": {
                "scan_depth": 2,
                "timeout": 10,
                "max_threads": 5,
                "rate_limit": 1.0,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            },
            "ui": {
                "update_interval": 500,
                "default_window_size": [900, 600],
                "theme": "system"
            },
            "reporting": {
                "default_report_format": "json",
                "default_report_directory": str(Path.home() / "analyzer_reports")
            },
            "logging": {
                "level": "INFO",
                "file": "analyzer.log"
            }
        }
        self.config = self.load()
        
    def load(self):
        """
        Charge la configuration à partir du fichier.
        
        Returns:
            dict: Configuration chargée ou configuration par défaut si le fichier n'existe pas
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # Fusionner avec la configuration par défaut pour garantir que tous les paramètres existent
                config = self.default_config.copy()
                self._deep_update(config, loaded_config)
                logger.info(f"Configuration chargée depuis {self.config_file}")
                return config
            else:
                logger.info(f"Fichier de configuration {self.config_file} non trouvé, utilisation des valeurs par défaut")
                return self.default_config.copy()
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            return self.default_config.copy()
    
    def save(self):
        """
        Enregistre la configuration actuelle dans le fichier.
        
        Returns:
            bool: True si l'enregistrement a réussi, False sinon
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration enregistrée dans {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la configuration: {str(e)}")
            return False
    
    def get(self, section, key=None, default=None):
        """
        Récupère une valeur de configuration.
        
        Args:
            section (str): Section de la configuration
            key (str, optional): Clé dans la section. Si None, retourne toute la section. Défaut: None
            default: Valeur par défaut à retourner si la clé n'existe pas
            
        Returns:
            La valeur de configuration ou la valeur par défaut
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def set(self, section, key, value):
        """
        Définit une valeur de configuration.
        
        Args:
            section (str): Section de la configuration
            key (str): Clé dans la section
            value: Valeur à définir
            
        Returns:
            bool: True si la valeur a été définie, False sinon
        """
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section][key] = value
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la définition de la configuration {section}.{key}: {str(e)}")
            return False
    
    def reset_to_default(self):
        """
        Réinitialise la configuration aux valeurs par défaut.
        
        Returns:
            dict: Configuration par défaut
        """
        self.config = self.default_config.copy()
        return self.config
    
    def _deep_update(self, target, source):
        """
        Met à jour un dictionnaire de façon récursive.
        
        Args:
            target (dict): Dictionnaire cible à mettre à jour
            source (dict): Dictionnaire source contenant les nouvelles valeurs
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
