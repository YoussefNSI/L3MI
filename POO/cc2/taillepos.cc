#include "taillepos.hh"

/* classe position */
void position_t::modifier_x(coord x) {
	_x = x;
}

void position_t::modifier_y(coord y) {
	_y = y;
}

position_t position_t::operator +(const vitesse_t& v){
    return position_t(x()+v.dx(), y()+v.dy());
}
/* classe vitesse */

void vitesse_t::modifier_dx(coord dx) {
	_dx = dx;
}

void vitesse_t::modifier_dy(coord dy) {
	_dy = dy;
}

vitesse_t vitesse_t::rebond_horizontal() const{
    return vitesse_t(dx(), -dy());
}

vitesse_t vitesse_t::rebond_vertical() const{
    return vitesse_t(-dx(), dy());
}

/* classe taille */

void taille_t::modifier_w(coord w) {
	_w = w;
}

void taille_t::modifier_h(coord h) {
	_h = h;
}


