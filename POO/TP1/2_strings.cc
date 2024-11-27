#include <iostream>
#include <string>

bool estunevoyelle(char c) {
	switch (c) {
		case 'a': case 'e': case 'i': case 'o': case 'u': case 'y':
		case 'A': case 'E': case 'I': case 'O': case 'U': case 'Y':
			return true;
		default:
			return false;
	}
}

unsigned int nbvoyelles(std::string const & s) {
	unsigned int result(0);
	for (char c : s)
		if (estunevoyelle(c))
			++result;
	return result;
}

// Version itérative
/*
bool palindrome(std::string const & s) {
    for (std::size_t i=0; i < (s.size()/2); ++i)
        if (s[i] != s[s.size()-1-i])
            return false;
    return true;
}
*/

// Deux version récursives :
// Version simple à comprendre, mais analysez aussi la version ci-dessous plus
// compacte et intéressante car il n'y a qu'une seule instruction (return)
// donc il est certain qu'on ne peut pas oublier de retourner une valeur.
/*
bool palindrome(std::string const & s) {
	if (s.size() <= 1)
		return true;
	else {
		if (s.front() != s.back())
			return false;
		else
			return palindrome(s.substr(1, s.size()-2));
	}
}
*/
bool palindrome(std::string const & s) {
	return (s.size() <= 1) || ((s.front() == s.back()) && palindrome(s.substr(1, s.size()-2)));
}

int main() {
	std::cout << nbvoyelles("bonjour") << std::endl;
	std::cout << palindrome("laval") << std::endl;
	std::cout << palindrome("lundi") << std::endl;
	std::cout << palindrome("entree") << std::endl;
	return 0;
}
