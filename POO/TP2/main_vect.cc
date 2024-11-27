#include <vector>
#include <iostream>
/*
std::vector<int> saisie() {
	std::vector<int> result;
	unsigned int nb;
	std::cout << "combien de valeurs ? "; std::cin >> nb;
	for (unsigned int i=0; i<nb; ++i) {
		std::cout << "valeur " << i << " ";
		int e;
		std::cin >> e;
		result.push_back(e);
	}
	return result;
}
*/
void saisie(std::vector<int> & s) {
    unsigned int nb=10;
	std::cout << "combien de valeurs ? "; std::cin >> nb;
	for (unsigned int i=0; i<nb; ++i) {
		std::cout << "valeur " << i << " ";
		int e;
		std::cin >> e;
		s.push_back(e);
	}
}

int maximum(std::vector<int> const & v) {
	int maxi(v[0]);
	for (std::vector<int>::size_type i=1; i<v.size(); ++i)
		if (v[i] > maxi)
			maxi = v[i];
	return maxi;
}
int maximumv2(std::vector<int> const & v) {
	int maxi(v[0]);
	for (auto i=v.begin(); i!=v.end(); ++i)
		if ((*i) > maxi)
			maxi = (*i);
	return maxi;
}
int maximumv3(std::vector<int> const & v) {
	int maxi(v[0]);
	for (int i : v)
		if (i > maxi)
			maxi = i;
	return maxi;
}

int main() {
    //std::vector<int> s(saisie()); // Pour la version qui retourne un vector.

    std::vector<int> s;
	saisie(s);
	std::cout << maximum(s) << std::endl;	

	return 0;
}
