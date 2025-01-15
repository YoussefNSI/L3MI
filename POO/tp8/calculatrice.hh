#include <QtWidgets>
class calcul: public QWidget {
	Q_OBJECT
	public:
		calcul();
	public slots:
		void oncliccalculer();
	private:
		QLineEdit * _operande1;
		QComboBox * _operateur;
		QLineEdit * _operande2;
		QLabel * _resultat;
		QPushButton * _calculer;
		QPushButton * _quitter;
};
