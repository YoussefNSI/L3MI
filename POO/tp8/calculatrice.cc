#include "calculatrice.hh"
#include <QString>

calcul::calcul()
	:QWidget(),
	 _operande1(new QLineEdit("", this)),
	 _operateur(new QComboBox(this)),
	 _operande2(new QLineEdit("", this)),
	 _resultat(new QLabel("", this)),
	 _calculer(new QPushButton("=", this)),
	 _quitter(new QPushButton("Quitter", this)) {
	resize(450,50);
	_operateur->addItem("+");
	_operateur->addItem("-");
	_operateur->addItem("*");
	_operateur->addItem("/");
	_operande1->setGeometry(10,10,70,30);
	_operateur->setGeometry(90,10,50,30);
	_operande2->setGeometry(150,10,70,30);
	_calculer->setGeometry(230,10,50,30);
	_resultat->setGeometry(290,10,70,30);
	_quitter->setGeometry(370,10,70,30);
	setWindowTitle("Calculatrice");
	connect(_calculer, &QPushButton::clicked, this, &calcul::oncliccalculer);
	connect(_quitter, &QPushButton::clicked, this, &calcul::close);
	connect(_operateur, &QComboBox::currentTextChanged, this, &calcul::oncliccalculer);
	connect(_operande1, &QLineEdit::textChanged, this, &calcul::oncliccalculer);
	connect(_operande2, &QLineEdit::textChanged, this, &calcul::oncliccalculer);
}

void calcul::oncliccalculer() {
	auto s1(_operande1->text().toStdString());
	auto s2(_operande2->text().toStdString());
	float f1(0), f2(0);
	if (!s1.empty())
		f1 = std::stof(s1);
	if (!s2.empty())
		f2 = std::stof(s2);
	float result(0);
	if (_operateur->currentText() == "+")
		result = f1+f2;
	else if (_operateur->currentText() == "-")
		result = f1-f2;
	else if (_operateur->currentText() == "*")
		result = f1*f2;
	else if (_operateur->currentText() == "/") {
		if (f2 == 0)
			result = 0;
		else
			result = f1/f2;
	}
	_resultat->setText(QString::fromStdString(std::to_string(result)));
}

int main(int argc, char *argv[]) {
	QApplication app(argc, argv);
	calcul f;
	f.show();
	return app.exec();
}
