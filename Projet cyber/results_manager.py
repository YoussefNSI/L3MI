"""
Module de gestion des résultats pour l'Analyseur de Sites Web.

Ce module fournit des fonctionnalités pour stocker, afficher et exporter
les résultats d'analyse.
"""

import json
import csv
import os
import logging
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QFileDialog
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor

logger = logging.getLogger('analyzer.results')

class ResultsManager(QObject):
    """
    Gère les résultats d'analyse.
    
    Cette classe fournit des méthodes pour stocker, afficher et exporter les
    résultats d'analyse.
    
    Signals:
        result_added (dict): Émis lorsqu'un résultat est ajouté
        results_cleared: Émis lorsque les résultats sont effacés
        
    Attributes:
        results (list): Liste des résultats
        results_table (QTableWidget): Tableau pour afficher les résultats
        details_widget (QTextEdit): Widget pour afficher les détails d'un résultat
        colors (dict): Couleurs pour les différents types de résultats
    """
    
    result_added = pyqtSignal(dict)
    results_cleared = pyqtSignal()
    
    def __init__(self, results_table, details_widget, colors=None):
        """
        Initialise le gestionnaire de résultats.
        
        Args:
            results_table (QTableWidget): Tableau pour afficher les résultats
            details_widget (QTextEdit): Widget pour afficher les détails d'un résultat
            colors (dict, optional): Couleurs pour les différents types de résultats
        """
        super().__init__()
        self.results = []
        self.results_table = results_table
        self.details_widget = details_widget
        self.colors = colors or {
            "vulnerability": QColor(255, 200, 200),
            "potential": QColor(255, 255, 200),
            "info": QColor(200, 255, 200),
            "warning": QColor(255, 235, 176)
        }
        logger.info("ResultsManager initialisé")

    def add_result(self, result_data):
        """
        Ajoute un résultat à la liste et au tableau.
        
        Args:
            result_data (dict): Données du résultat
            
        Returns:
            int: Index du résultat ajouté
        """
        try:
            # Ajouter aux données stockées
            self.results.append(result_data)
            result_index = len(self.results) - 1

            # Ajouter une nouvelle ligne au tableau
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            # Remplir les cellules
            self.results_table.setItem(row, 0, QTableWidgetItem(result_data.get('type', 'Inconnu')))
            self.results_table.setItem(row, 1, QTableWidgetItem(result_data.get('context', 'Inconnu')))

            # Combiner URL et payload pour la troisième colonne
            payload_info = result_data.get('url', '')
            if result_data.get('payload'):
                if isinstance(result_data['payload'], list):
                    payload_preview = str(result_data['payload'][0])[:30]
                else:
                    payload_preview = str(result_data['payload'])[:30]
                
                if len(str(result_data['payload'])) > 30:
                    payload_preview += "..."
                payload_info += f" - {payload_preview}"

            self.results_table.setItem(row, 2, QTableWidgetItem(payload_info))

            # Colorer la ligne en fonction du type de résultat
            result_type = result_data.get('type', 'info')
            color = self.colors.get(result_type, QColor(255, 255, 255))
            
            for col in range(3):
                self.results_table.item(row, col).setBackground(color)

            self.results_table.selectRow(row)
            self.results_table.scrollToBottom()
            
            # Émettre le signal
            self.result_added.emit(result_data)
            
            logger.debug(f"Résultat ajouté: {result_data.get('context')} - {payload_preview}")
            
            return result_index
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout d'un résultat: {str(e)}")
            raise

    def clear_results(self):
        """
        Efface tous les résultats.
        """
        self.results = []
        self.results_table.setRowCount(0)
        self.details_widget.clear()
        self.results_cleared.emit()
        logger.info("Résultats effacés")

    def show_result_details(self, row):
        """
        Affiche les détails d'un résultat.
        
        Args:
            row (int): Index de la ligne dans le tableau
        """
        if row < 0 or row >= len(self.results):
            return
            
        result = self.results[row]
        details = (f"# Détails du résultat\n\n"
                  f"- **Type:** {result.get('type', 'Inconnu')}\n"
                  f"- **Contexte:** {result.get('context', 'Inconnu')}\n"
                  f"- **URL:** {result.get('url', 'N/A')}\n\n"
                  f"## Charge utile\n\n```\n{result.get('payload', 'N/A')}\n"
                  f"```")
        self.details_widget.setMarkdown(details)
        logger.debug(f"Détails affichés pour le résultat {row}")

    def save_results(self, parent_widget):
        """
        Enregistre les résultats dans un fichier.
        
        Args:
            parent_widget: Widget parent pour les boîtes de dialogue
            
        Returns:
            bool: True si l'enregistrement a réussi, False sinon
        """
        try:
            if not self.results:
                QMessageBox.information(parent_widget, "Information", "Aucun résultat à enregistrer.")
                return False

            # Demander le format
            formats = {"json": "JSON (*.json)", "csv": "CSV (*.csv)", "txt": "Texte (*.txt)"}
            format_dialog = QFileDialog(parent_widget)
            format_dialog.setFileMode(QFileDialog.AnyFile)
            format_dialog.setAcceptMode(QFileDialog.AcceptSave)
            format_dialog.setNameFilters(list(formats.values()))
            
            if not format_dialog.exec_():
                return False
                
            selected_filter = format_dialog.selectedNameFilter()
            file_format = next((k for k, v in formats.items() if v == selected_filter), "txt")
            
            file_path = format_dialog.selectedFiles()[0]
            if not file_path:
                return False
                
            # Ajouter l'extension si nécessaire
            if not any(file_path.endswith(ext) for ext in ['.json', '.csv', '.txt']):
                file_path += f".{file_format}"
                
            # Exporter selon le format
            if file_format == "json":
                self._save_json(file_path)
            elif file_format == "csv":
                self._save_csv(file_path)
            else:
                self._save_txt(file_path)
                
            logger.info(f"Résultats enregistrés dans {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement des résultats: {str(e)}")
            QMessageBox.warning(parent_widget, "Erreur d'enregistrement", 
                                f"Impossible d'enregistrer les résultats: {str(e)}")
            return False
            
    def _save_json(self, file_path):
        """
        Enregistre les résultats au format JSON.
        
        Args:
            file_path (str): Chemin du fichier
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            export_data = {
                "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results_count": len(self.results),
                "results": self.results
            }
            json.dump(export_data, f, indent=2)
            
    def _save_csv(self, file_path):
        """
        Enregistre les résultats au format CSV.
        
        Args:
            file_path (str): Chemin du fichier
        """
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "Contexte", "URL", "Payload"])
            
            for result in self.results:
                writer.writerow([
                    result.get('type', 'Inconnu'),
                    result.get('context', 'Inconnu'),
                    result.get('url', 'N/A'),
                    str(result.get('payload', 'N/A'))
                ])
                
    def _save_txt(self, file_path):
        """
        Enregistre les résultats au format texte.
        
        Args:
            file_path (str): Chemin du fichier
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Rapport d'analyse - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Nombre de résultats: {len(self.results)}\n\n")
            
            for i, result in enumerate(self.results):
                f.write(f"Résultat #{i+1}\n")
                f.write(f"Type: {result.get('type', 'Inconnu')}\n")
                f.write(f"Contexte: {result.get('context', 'Inconnu')}\n")
                f.write(f"URL: {result.get('url', 'N/A')}\n")
                f.write(f"Payload: {result.get('payload', 'N/A')}\n")
                f.write("\n")
    
    def generate_summary(self):
        """
        Génère un résumé des résultats d'analyse.
        
        Returns:
            str: Résumé au format Markdown
        """
        vuln_count = sum(1 for r in self.results if r.get('type') == 'vulnerability')
        
        summary = (f"# Résumé de l'analyse\n\n"
                   f"- **Date et heure:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"- **Résultats trouvés:** {len(self.results)}\n"
                   f"- **Vulnérabilités détectées:** {vuln_count}\n\n"
                   f"## Détails des résultats\n\n")

        for i, result in enumerate(self.results):
            summary += (f"### Résultat #{i + 1}\n"
                        f"- **Type:** {result.get('type', 'Inconnu')}\n"
                        f"- **Contexte:** {result.get('context', 'Inconnu')}\n"
                        f"- **URL:** {result.get('url', 'N/A')}\n"
                        f"- **Payload:** {result.get('payload', 'N/A')}\n\n")
                        
        return summary
