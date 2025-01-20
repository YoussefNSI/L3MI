#include <QtWidgets>
#include "sequence.hh"
class simon: public QWidget {
	Q_OBJECT
	public:
		simon(sequence const & s);
	public slots:
		void onclicquitter();
		void oncliccouleur();
	private:
		couleur boutonverscouleur(QPushButton const * b) const;
		void perdu();
		void recommencer();
		void activerjoueur(unsigned char j);
	private:
		std::map<couleur, QPushButton *> _boutonscouleurs;
		QPushButton * _quitter;
		std::array<QLineEdit *, 2> _joueurs;
		std::array<QLabel *, 2> _joueursmarque;
		enum class etat {
			enregistrement,
			restitution,
		} _etat;
		sequence::indice _courant;
		sequence _sequence;
		unsigned char _joueuractuel;
};
