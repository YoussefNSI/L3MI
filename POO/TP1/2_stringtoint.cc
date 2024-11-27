#include <iostream>
#include <string>

unsigned int chaineversentier(std::string const & s) {
	if (s.empty())
		return 0;
	else
		return chaineversentier(s.substr(0, s.size()-1))*10 + (s.back()-'0');
}

int main() {
	std::cout << chaineversentier("2042") << std::endl;
	return 0;
}
