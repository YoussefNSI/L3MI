#pragma once
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

using coord = signed short;

class position {
public:
	position(coord x, coord y)
	    :_x(x), _y(y) {}
	position(position const & p) =default;
	position & operator=(position const & p) =default; // ou le définir
	coord x() const {
		return _x;
	}
	coord y() const {
		return _y;
	}
	void setx(coord x) {
		assert((x<8) && (x>=0));
		_x = x;
	}
	void sety(coord y) {
		assert((y<8) && (y>=0));
		_y = y;
	}
	bool operator==(position const & p) const {
		return (_x == p._x) && (_y == p._y);
	}
	bool operator!=(position const & p) const {
		return !operator==(p);
	}
	bool estvalide() const {
		return ((_x >= 0) && (_x < 8) && (_y >= 0) && (_y < 8));
	}
private:
	coord _x;
	coord _y;
};

std::ostream & operator<<(std::ostream & os, position const & p);

enum class couleur {
	noir,
	blanc
};

class piece {
public:
	piece(position const & p, couleur c)
		:_position(p), _couleur(c) {
	}
	piece(piece const& piece) =default;
	virtual std::unique_ptr<piece> clone() const =0;
	virtual ~piece();
	virtual unsigned int valeur() const =0;
	virtual char symbole() const =0;
	position const & pos() const {
		return _position;
	}
	couleur coul() const {
		return _couleur;
	}
	std::string tostring() const {
		return std::string() + symbole() + ((_couleur == couleur::noir) ? "N" : "B") + std::to_string(_position.x()) + std::to_string(_position.y());
	}
	virtual std::vector<position> deplacementspossibles() const =0;
	bool accepterposition(position const & p) const;
	bool deplacer(position const & p);
private:
	position _position;
	couleur _couleur;
};

class pion: public piece {
public:
	pion(position const & p, couleur c)
		:piece(p, c) {}
	std::unique_ptr<piece> clone() const override {
		return std::make_unique<pion>(*this);
	}
	unsigned int valeur() const override {
		return 1;
	}
	char symbole() const override {
		return 'P';
	}
	std::vector<position> deplacementspossibles() const override;
};

class roi: public piece {
public:
	roi(position const & p, couleur c)
		:piece(p, c) {}
	std::unique_ptr<piece> clone() const override {
		return std::make_unique<roi>(*this);
	}
	unsigned int valeur() const override {
		return 20;
	}
	char symbole() const override {
		return 'R';
	}
	std::vector<position> deplacementspossibles() const override;
};

class dame: public piece {
public:
	dame(position const & p, couleur c)
		:piece(p, c) {}
	std::unique_ptr<piece> clone() const override {
		return std::make_unique<dame>(*this);
	}
	unsigned int valeur() const override {
		return 9;
	}
	char symbole() const override {
		return 'D';
	}
	std::vector<position> deplacementspossibles() const override;
};

class cavalier: public piece {
public:
	cavalier(position const & p, couleur c)
		:piece(p, c) {}
	std::unique_ptr<piece> clone() const override {
		return std::make_unique<cavalier>(*this);
	}
	unsigned int valeur() const override {
		return 3;
	}
	char symbole() const override {
		return 'C';
	}
	std::vector<position> deplacementspossibles() const override;
};
