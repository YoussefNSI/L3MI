%skeleton "lalr1.cc"
%require "3.0"

%defines
%define api.parser.class { Parser }
%define api.value.type variant
%define parse.assert

%locations

%code requires{
    class Scanner;
    class Driver;
}

%parse-param { Scanner &scanner }
%parse-param { Driver &driver }

%code{
    #include <iostream>
    #include <string>
    #include <cmath>
    
    #include "scanner.hh"
    #include "driver.hh"

    #undef  yylex
    #define yylex scanner.yylex

    int somme=0;
}

%token           NL
%token           END
%token<int>      NUMBER
%token           RAC

%left '+' '-'
%left '*' '/'

%type<int>      expression

%precedence NEG

%%

programme:
    expression NL {
        std::cout << "#-> " << $1 << std::endl;
    } programme
    | END NL {
        YYACCEPT;
    }

expression:
        NUMBER
    |   expression '+' expression {
        $$ = $1 + $3;
    }
    |   expression '*' expression {
        $$ = $1 * $3;
    }
    |   expression '-' expression {
        $$ = $1 - $3;
    }
    |   expression '/' expression {
        $$ = $1 / $3;
    }
    | '(' expression ')' {
        $$ = $2;
    }
    | expression '^' expression {
        $$ = pow($1, $3);
    }
    | '-' expression %prec NEG {
        $$ = -$2;
    }
    | RAC expression {
        $$ = sqrt($2);
    }
%%

void yy::Parser::error( const location_type &l, const std::string & err_msg) {
    std::cerr << "Erreur : " << l << ", " << err_msg << std::endl;
}


