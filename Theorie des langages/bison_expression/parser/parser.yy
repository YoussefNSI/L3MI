%skeleton "lalr1.cc"
%require "3.0"

%defines
%define api.parser.class { Parser }
%define api.value.type variant
%define parse.assert

%locations

%code requires{
    #include "contexte.hh"
    #include "expressionBinaire.hh"
    #include "expressionUnaire.hh"
    #include "constante.hh"
    #include "variable.hh"

    class Scanner;
    class Driver;
}

%parse-param { Scanner &scanner }
%parse-param { Driver &driver }

%code{
    #include <iostream>
    #include <string>
    
    #include "scanner.hh"
    #include "driver.hh"

    #undef  yylex
    #define yylex scanner.yylex
}

%token                  NL
%token                  END
%token <double>            NUMBER
%token<std::string>      IDENT

%type <ExpressionPtr>             operation
%left '-' '+'
%left '*' '/'
%precedence  NEG

%%

programme:
    instruction NL programme
    | END NL {
        YYACCEPT;
    }

instruction:
    expression  {
        YYACCEPT;
    }
    | affectation {
      YYACCEPT;
    }

expression:
    operation {
        try {
            double val = $1->calculer(driver.getContexte());
            std::cout << "#-> " << val << std::endl;
        } catch(const std::exception& err) {
            std::cerr << "#-> " << err.what() << std::endl;
        }
    }

affectation:
    IDENT '=' operation {
        try {
            double val = $3->calculer(driver.getContexte());
            driver.setVariable($1, val);
            std::cout << "#-> " << $1 << " = " << val << std::endl;
        } catch(const std::exception& err) {
            std::cerr << "#-> " << err.what() << std::endl;
        }
    }

operation:
    NUMBER {
        $$ = std::make_shared<Constante>($1);
    }
    | IDENT {
        $$ = std::make_shared<Variable>($1);
    }
    | '(' operation ')' {
        $$ = $2;
    }
    | operation '+' operation {
        $$ = std::make_shared<ExpressionBinaire>($1, $3, OperateurBinaire::plus);
    }
    | operation '-' operation {
        $$ = std::make_shared<ExpressionBinaire>($1, $3, OperateurBinaire::moins);
    }
    | operation '*' operation {
        $$ = std::make_shared<ExpressionBinaire>($1, $3, OperateurBinaire::multiplie);
    }
    | operation '/' operation {
        $$ = std::make_shared<ExpressionBinaire>($1, $3, OperateurBinaire::divise);
    }
    | '-' operation %prec NEG {
        $$ = std::make_shared<ExpressionUnaire>($2, OperateurUnaire::neg);
    }

%%

void yy::Parser::error( const location_type &l, const std::string & err_msg) {
    std::cerr << "Erreur : " << l << ", " << err_msg << std::endl;
}
