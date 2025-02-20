#include "parser.hpp"
#include "scanner.hh"
#include "driver.hh"

#include <iostream>
#include <fstream>
#include <variant>
#include <vector>
#include <map>
#include <string>

#include <cstring>

Document* doc = new Document();

int main( int  argc, char* argv[]) {
    Driver * driver = new Driver;
    Scanner * scanner = new Scanner(std::cin, std::cout);
    yy::Parser * parser = new yy::Parser(*scanner, *driver);

    parser->parse();
    doc->afficherBlocs();

    return 0;
}
