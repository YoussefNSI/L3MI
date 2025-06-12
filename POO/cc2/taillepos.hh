#pragma once

using coord = float;

class taille_t {
	public:
	taille_t(coord w, coord h)
		:_w(w), _h(h) {}
	coord w() const { return _w; }
	coord h() const { return _h; }
	void modifier_w(coord w);
	void modifier_h(coord h);
	private:
	coord _w;
	coord _h;
};

class vitesse_t {
	public:
	vitesse_t(coord dx, coord dy)
		:_dx(dx), _dy(dy) {}
	coord dx() const { return _dx; }
	coord dy() const { return _dy; }
	void modifier_dx(coord dx);
	void modifier_dy(coord dy);
    vitesse_t rebond_horizontal() const;
    vitesse_t rebond_vertical() const;
	private:
	coord _dx;
	coord _dy;
};

class position_t {
    public:
    position_t(coord x, coord y)
        :_x(x), _y(y) {}
    coord x() const { return _x; }
    coord y() const { return _y; }
    void modifier_x(coord x);
    void modifier_y(coord y);
    position_t operator+(const vitesse_t& v);
    private:
    coord _x;
    coord _y;
};
