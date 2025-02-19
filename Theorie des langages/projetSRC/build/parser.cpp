// A Bison parser, made by GNU Bison 3.8.2.

// Skeleton implementation for Bison LALR(1) parsers in C++

// Copyright (C) 2002-2015, 2018-2021 Free Software Foundation, Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// As a special exception, you may create a larger work that contains
// part or all of the Bison parser skeleton and distribute that work
// under terms of your choice, so long as that work isn't itself a
// parser generator using the skeleton or a modified version thereof
// as a parser skeleton.  Alternatively, if you modify or redistribute
// the parser skeleton itself, you may (at your option) remove this
// special exception, which will cause the skeleton and the resulting
// Bison output files to be licensed under the GNU General Public
// License without this special exception.

// This special exception was added by the Free Software Foundation in
// version 2.2 of Bison.

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.





#include "parser.hpp"


// Unqualified %code blocks.
#line 34 "parser/parser.yy"

    #include <iostream>
    #include <string>
    #include <memory>
    #include <map>
    #include <variant>
    
    #include "scanner.hh"
    #include "driver.hh"
    #include "bloc.h"


    std::unique_ptr<Document> doc = std::make_unique<Document>();


    #undef  yylex
    #define yylex scanner.yylex

#line 65 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"


#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> // FIXME: INFRINGES ON USER NAME SPACE.
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif


// Whether we are compiled with exception support.
#ifndef YY_EXCEPTIONS
# if defined __GNUC__ && !defined __EXCEPTIONS
#  define YY_EXCEPTIONS 0
# else
#  define YY_EXCEPTIONS 1
# endif
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K].location)
/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

# ifndef YYLLOC_DEFAULT
#  define YYLLOC_DEFAULT(Current, Rhs, N)                               \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).begin  = YYRHSLOC (Rhs, 1).begin;                   \
          (Current).end    = YYRHSLOC (Rhs, N).end;                     \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).begin = (Current).end = YYRHSLOC (Rhs, 0).end;      \
        }                                                               \
    while (false)
# endif


// Enable debugging if requested.
#if YYDEBUG

// A pseudo ostream that takes yydebug_ into account.
# define YYCDEBUG if (yydebug_) (*yycdebug_)

# define YY_SYMBOL_PRINT(Title, Symbol)         \
  do {                                          \
    if (yydebug_)                               \
    {                                           \
      *yycdebug_ << Title << ' ';               \
      yy_print_ (*yycdebug_, Symbol);           \
      *yycdebug_ << '\n';                       \
    }                                           \
  } while (false)

# define YY_REDUCE_PRINT(Rule)          \
  do {                                  \
    if (yydebug_)                       \
      yy_reduce_print_ (Rule);          \
  } while (false)

# define YY_STACK_PRINT()               \
  do {                                  \
    if (yydebug_)                       \
      yy_stack_print_ ();                \
  } while (false)

#else // !YYDEBUG

# define YYCDEBUG if (false) std::cerr
# define YY_SYMBOL_PRINT(Title, Symbol)  YY_USE (Symbol)
# define YY_REDUCE_PRINT(Rule)           static_cast<void> (0)
# define YY_STACK_PRINT()                static_cast<void> (0)

#endif // !YYDEBUG

#define yyerrok         (yyerrstatus_ = 0)
#define yyclearin       (yyla.clear ())

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYRECOVERING()  (!!yyerrstatus_)

namespace yy {
#line 157 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"

  /// Build a parser object.
   Parser :: Parser  (Scanner &scanner_yyarg, Driver &driver_yyarg)
#if YYDEBUG
    : yydebug_ (false),
      yycdebug_ (&std::cerr),
#else
    :
#endif
      scanner (scanner_yyarg),
      driver (driver_yyarg)
  {}

   Parser ::~ Parser  ()
  {}

   Parser ::syntax_error::~syntax_error () YY_NOEXCEPT YY_NOTHROW
  {}

  /*---------.
  | symbol.  |
  `---------*/

  // basic_symbol.
  template <typename Base>
   Parser ::basic_symbol<Base>::basic_symbol (const basic_symbol& that)
    : Base (that)
    , value ()
    , location (that.location)
  {
    switch (this->kind ())
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.copy< TitreInfo > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
        value.copy< int > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_attributs: // attributs
        value.copy< std::map<std::string, std::map<std::string, std::string>> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.copy< std::map<std::string, std::string> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_PARAGRAPHE: // PARAGRAPHE
      case symbol_kind::S_IMAGE: // IMAGE
      case symbol_kind::S_DEFINE: // DEFINE
      case symbol_kind::S_TITREPAGE: // TITREPAGE
      case symbol_kind::S_STYLE: // STYLE
      case symbol_kind::S_ATTRIBUT: // ATTRIBUT
      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_EGAL: // EGAL
      case symbol_kind::S_CROCHET_FERMANT: // CROCHET_FERMANT
      case symbol_kind::S_CROCHET_OUVRANT: // CROCHET_OUVRANT
      case symbol_kind::S_DEUX_POINTS: // DEUX_POINTS
      case symbol_kind::S_VIRGULE: // VIRGULE
      case symbol_kind::S_POINT_VIRGULE: // POINT_VIRGULE
      case symbol_kind::S_PARENTHESE_OUVRANTE: // PARENTHESE_OUVRANTE
      case symbol_kind::S_PARENTHESE_FERMANTE: // PARENTHESE_FERMANTE
      case symbol_kind::S_ACCOLADE_OUVRANTE: // ACCOLADE_OUVRANTE
      case symbol_kind::S_ACCOLADE_FERMANTE: // ACCOLADE_FERMANTE
      case symbol_kind::S_LARGEUR: // LARGEUR
      case symbol_kind::S_HAUTEUR: // HAUTEUR
      case symbol_kind::S_COULEURTEXTE: // COULEURTEXTE
      case symbol_kind::S_COULEURFOND: // COULEURFOND
      case symbol_kind::S_OPACITE: // OPACITE
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.copy< std::string* > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_element: // element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.copy< std::variant<int, std::string, std::unique_ptr<Bloc>> > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

  }




  template <typename Base>
   Parser ::symbol_kind_type
   Parser ::basic_symbol<Base>::type_get () const YY_NOEXCEPT
  {
    return this->kind ();
  }


  template <typename Base>
  bool
   Parser ::basic_symbol<Base>::empty () const YY_NOEXCEPT
  {
    return this->kind () == symbol_kind::S_YYEMPTY;
  }

  template <typename Base>
  void
   Parser ::basic_symbol<Base>::move (basic_symbol& s)
  {
    super_type::move (s);
    switch (this->kind ())
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.move< TitreInfo > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
        value.move< int > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_attributs: // attributs
        value.move< std::map<std::string, std::map<std::string, std::string>> > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.move< std::map<std::string, std::string> > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_PARAGRAPHE: // PARAGRAPHE
      case symbol_kind::S_IMAGE: // IMAGE
      case symbol_kind::S_DEFINE: // DEFINE
      case symbol_kind::S_TITREPAGE: // TITREPAGE
      case symbol_kind::S_STYLE: // STYLE
      case symbol_kind::S_ATTRIBUT: // ATTRIBUT
      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_EGAL: // EGAL
      case symbol_kind::S_CROCHET_FERMANT: // CROCHET_FERMANT
      case symbol_kind::S_CROCHET_OUVRANT: // CROCHET_OUVRANT
      case symbol_kind::S_DEUX_POINTS: // DEUX_POINTS
      case symbol_kind::S_VIRGULE: // VIRGULE
      case symbol_kind::S_POINT_VIRGULE: // POINT_VIRGULE
      case symbol_kind::S_PARENTHESE_OUVRANTE: // PARENTHESE_OUVRANTE
      case symbol_kind::S_PARENTHESE_FERMANTE: // PARENTHESE_FERMANTE
      case symbol_kind::S_ACCOLADE_OUVRANTE: // ACCOLADE_OUVRANTE
      case symbol_kind::S_ACCOLADE_FERMANTE: // ACCOLADE_FERMANTE
      case symbol_kind::S_LARGEUR: // LARGEUR
      case symbol_kind::S_HAUTEUR: // HAUTEUR
      case symbol_kind::S_COULEURTEXTE: // COULEURTEXTE
      case symbol_kind::S_COULEURFOND: // COULEURFOND
      case symbol_kind::S_OPACITE: // OPACITE
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.move< std::string* > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_element: // element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.move< std::variant<int, std::string, std::unique_ptr<Bloc>> > (YY_MOVE (s.value));
        break;

      default:
        break;
    }

    location = YY_MOVE (s.location);
  }

  // by_kind.
   Parser ::by_kind::by_kind () YY_NOEXCEPT
    : kind_ (symbol_kind::S_YYEMPTY)
  {}

#if 201103L <= YY_CPLUSPLUS
   Parser ::by_kind::by_kind (by_kind&& that) YY_NOEXCEPT
    : kind_ (that.kind_)
  {
    that.clear ();
  }
#endif

   Parser ::by_kind::by_kind (const by_kind& that) YY_NOEXCEPT
    : kind_ (that.kind_)
  {}

   Parser ::by_kind::by_kind (token_kind_type t) YY_NOEXCEPT
    : kind_ (yytranslate_ (t))
  {}



  void
   Parser ::by_kind::clear () YY_NOEXCEPT
  {
    kind_ = symbol_kind::S_YYEMPTY;
  }

  void
   Parser ::by_kind::move (by_kind& that)
  {
    kind_ = that.kind_;
    that.clear ();
  }

   Parser ::symbol_kind_type
   Parser ::by_kind::kind () const YY_NOEXCEPT
  {
    return kind_;
  }


   Parser ::symbol_kind_type
   Parser ::by_kind::type_get () const YY_NOEXCEPT
  {
    return this->kind ();
  }



  // by_state.
   Parser ::by_state::by_state () YY_NOEXCEPT
    : state (empty_state)
  {}

   Parser ::by_state::by_state (const by_state& that) YY_NOEXCEPT
    : state (that.state)
  {}

  void
   Parser ::by_state::clear () YY_NOEXCEPT
  {
    state = empty_state;
  }

  void
   Parser ::by_state::move (by_state& that)
  {
    state = that.state;
    that.clear ();
  }

   Parser ::by_state::by_state (state_type s) YY_NOEXCEPT
    : state (s)
  {}

   Parser ::symbol_kind_type
   Parser ::by_state::kind () const YY_NOEXCEPT
  {
    if (state == empty_state)
      return symbol_kind::S_YYEMPTY;
    else
      return YY_CAST (symbol_kind_type, yystos_[+state]);
  }

   Parser ::stack_symbol_type::stack_symbol_type ()
  {}

   Parser ::stack_symbol_type::stack_symbol_type (YY_RVREF (stack_symbol_type) that)
    : super_type (YY_MOVE (that.state), YY_MOVE (that.location))
  {
    switch (that.kind ())
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.YY_MOVE_OR_COPY< TitreInfo > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
        value.YY_MOVE_OR_COPY< int > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_attributs: // attributs
        value.YY_MOVE_OR_COPY< std::map<std::string, std::map<std::string, std::string>> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.YY_MOVE_OR_COPY< std::map<std::string, std::string> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_PARAGRAPHE: // PARAGRAPHE
      case symbol_kind::S_IMAGE: // IMAGE
      case symbol_kind::S_DEFINE: // DEFINE
      case symbol_kind::S_TITREPAGE: // TITREPAGE
      case symbol_kind::S_STYLE: // STYLE
      case symbol_kind::S_ATTRIBUT: // ATTRIBUT
      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_EGAL: // EGAL
      case symbol_kind::S_CROCHET_FERMANT: // CROCHET_FERMANT
      case symbol_kind::S_CROCHET_OUVRANT: // CROCHET_OUVRANT
      case symbol_kind::S_DEUX_POINTS: // DEUX_POINTS
      case symbol_kind::S_VIRGULE: // VIRGULE
      case symbol_kind::S_POINT_VIRGULE: // POINT_VIRGULE
      case symbol_kind::S_PARENTHESE_OUVRANTE: // PARENTHESE_OUVRANTE
      case symbol_kind::S_PARENTHESE_FERMANTE: // PARENTHESE_FERMANTE
      case symbol_kind::S_ACCOLADE_OUVRANTE: // ACCOLADE_OUVRANTE
      case symbol_kind::S_ACCOLADE_FERMANTE: // ACCOLADE_FERMANTE
      case symbol_kind::S_LARGEUR: // LARGEUR
      case symbol_kind::S_HAUTEUR: // HAUTEUR
      case symbol_kind::S_COULEURTEXTE: // COULEURTEXTE
      case symbol_kind::S_COULEURFOND: // COULEURFOND
      case symbol_kind::S_OPACITE: // OPACITE
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.YY_MOVE_OR_COPY< std::string* > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_element: // element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.YY_MOVE_OR_COPY< std::variant<int, std::string, std::unique_ptr<Bloc>> > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

#if 201103L <= YY_CPLUSPLUS
    // that is emptied.
    that.state = empty_state;
#endif
  }

   Parser ::stack_symbol_type::stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) that)
    : super_type (s, YY_MOVE (that.location))
  {
    switch (that.kind ())
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.move< TitreInfo > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
        value.move< int > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_attributs: // attributs
        value.move< std::map<std::string, std::map<std::string, std::string>> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.move< std::map<std::string, std::string> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_PARAGRAPHE: // PARAGRAPHE
      case symbol_kind::S_IMAGE: // IMAGE
      case symbol_kind::S_DEFINE: // DEFINE
      case symbol_kind::S_TITREPAGE: // TITREPAGE
      case symbol_kind::S_STYLE: // STYLE
      case symbol_kind::S_ATTRIBUT: // ATTRIBUT
      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_EGAL: // EGAL
      case symbol_kind::S_CROCHET_FERMANT: // CROCHET_FERMANT
      case symbol_kind::S_CROCHET_OUVRANT: // CROCHET_OUVRANT
      case symbol_kind::S_DEUX_POINTS: // DEUX_POINTS
      case symbol_kind::S_VIRGULE: // VIRGULE
      case symbol_kind::S_POINT_VIRGULE: // POINT_VIRGULE
      case symbol_kind::S_PARENTHESE_OUVRANTE: // PARENTHESE_OUVRANTE
      case symbol_kind::S_PARENTHESE_FERMANTE: // PARENTHESE_FERMANTE
      case symbol_kind::S_ACCOLADE_OUVRANTE: // ACCOLADE_OUVRANTE
      case symbol_kind::S_ACCOLADE_FERMANTE: // ACCOLADE_FERMANTE
      case symbol_kind::S_LARGEUR: // LARGEUR
      case symbol_kind::S_HAUTEUR: // HAUTEUR
      case symbol_kind::S_COULEURTEXTE: // COULEURTEXTE
      case symbol_kind::S_COULEURFOND: // COULEURFOND
      case symbol_kind::S_OPACITE: // OPACITE
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.move< std::string* > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_element: // element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.move< std::variant<int, std::string, std::unique_ptr<Bloc>> > (YY_MOVE (that.value));
        break;

      default:
        break;
    }

    // that is emptied.
    that.kind_ = symbol_kind::S_YYEMPTY;
  }

#if YY_CPLUSPLUS < 201103L
   Parser ::stack_symbol_type&
   Parser ::stack_symbol_type::operator= (const stack_symbol_type& that)
  {
    state = that.state;
    switch (that.kind ())
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.copy< TitreInfo > (that.value);
        break;

      case symbol_kind::S_ENTIER: // ENTIER
        value.copy< int > (that.value);
        break;

      case symbol_kind::S_attributs: // attributs
        value.copy< std::map<std::string, std::map<std::string, std::string>> > (that.value);
        break;

      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.copy< std::map<std::string, std::string> > (that.value);
        break;

      case symbol_kind::S_PARAGRAPHE: // PARAGRAPHE
      case symbol_kind::S_IMAGE: // IMAGE
      case symbol_kind::S_DEFINE: // DEFINE
      case symbol_kind::S_TITREPAGE: // TITREPAGE
      case symbol_kind::S_STYLE: // STYLE
      case symbol_kind::S_ATTRIBUT: // ATTRIBUT
      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_EGAL: // EGAL
      case symbol_kind::S_CROCHET_FERMANT: // CROCHET_FERMANT
      case symbol_kind::S_CROCHET_OUVRANT: // CROCHET_OUVRANT
      case symbol_kind::S_DEUX_POINTS: // DEUX_POINTS
      case symbol_kind::S_VIRGULE: // VIRGULE
      case symbol_kind::S_POINT_VIRGULE: // POINT_VIRGULE
      case symbol_kind::S_PARENTHESE_OUVRANTE: // PARENTHESE_OUVRANTE
      case symbol_kind::S_PARENTHESE_FERMANTE: // PARENTHESE_FERMANTE
      case symbol_kind::S_ACCOLADE_OUVRANTE: // ACCOLADE_OUVRANTE
      case symbol_kind::S_ACCOLADE_FERMANTE: // ACCOLADE_FERMANTE
      case symbol_kind::S_LARGEUR: // LARGEUR
      case symbol_kind::S_HAUTEUR: // HAUTEUR
      case symbol_kind::S_COULEURTEXTE: // COULEURTEXTE
      case symbol_kind::S_COULEURFOND: // COULEURFOND
      case symbol_kind::S_OPACITE: // OPACITE
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.copy< std::string* > (that.value);
        break;

      case symbol_kind::S_element: // element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.copy< std::variant<int, std::string, std::unique_ptr<Bloc>> > (that.value);
        break;

      default:
        break;
    }

    location = that.location;
    return *this;
  }

   Parser ::stack_symbol_type&
   Parser ::stack_symbol_type::operator= (stack_symbol_type& that)
  {
    state = that.state;
    switch (that.kind ())
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.move< TitreInfo > (that.value);
        break;

      case symbol_kind::S_ENTIER: // ENTIER
        value.move< int > (that.value);
        break;

      case symbol_kind::S_attributs: // attributs
        value.move< std::map<std::string, std::map<std::string, std::string>> > (that.value);
        break;

      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.move< std::map<std::string, std::string> > (that.value);
        break;

      case symbol_kind::S_PARAGRAPHE: // PARAGRAPHE
      case symbol_kind::S_IMAGE: // IMAGE
      case symbol_kind::S_DEFINE: // DEFINE
      case symbol_kind::S_TITREPAGE: // TITREPAGE
      case symbol_kind::S_STYLE: // STYLE
      case symbol_kind::S_ATTRIBUT: // ATTRIBUT
      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_EGAL: // EGAL
      case symbol_kind::S_CROCHET_FERMANT: // CROCHET_FERMANT
      case symbol_kind::S_CROCHET_OUVRANT: // CROCHET_OUVRANT
      case symbol_kind::S_DEUX_POINTS: // DEUX_POINTS
      case symbol_kind::S_VIRGULE: // VIRGULE
      case symbol_kind::S_POINT_VIRGULE: // POINT_VIRGULE
      case symbol_kind::S_PARENTHESE_OUVRANTE: // PARENTHESE_OUVRANTE
      case symbol_kind::S_PARENTHESE_FERMANTE: // PARENTHESE_FERMANTE
      case symbol_kind::S_ACCOLADE_OUVRANTE: // ACCOLADE_OUVRANTE
      case symbol_kind::S_ACCOLADE_FERMANTE: // ACCOLADE_FERMANTE
      case symbol_kind::S_LARGEUR: // LARGEUR
      case symbol_kind::S_HAUTEUR: // HAUTEUR
      case symbol_kind::S_COULEURTEXTE: // COULEURTEXTE
      case symbol_kind::S_COULEURFOND: // COULEURFOND
      case symbol_kind::S_OPACITE: // OPACITE
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.move< std::string* > (that.value);
        break;

      case symbol_kind::S_element: // element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.move< std::variant<int, std::string, std::unique_ptr<Bloc>> > (that.value);
        break;

      default:
        break;
    }

    location = that.location;
    // that is emptied.
    that.state = empty_state;
    return *this;
  }
#endif

  template <typename Base>
  void
   Parser ::yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const
  {
    if (yymsg)
      YY_SYMBOL_PRINT (yymsg, yysym);
  }

#if YYDEBUG
  template <typename Base>
  void
   Parser ::yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const
  {
    std::ostream& yyoutput = yyo;
    YY_USE (yyoutput);
    if (yysym.empty ())
      yyo << "empty symbol";
    else
      {
        symbol_kind_type yykind = yysym.kind ();
        yyo << (yykind < YYNTOKENS ? "token" : "nterm")
            << ' ' << yysym.name () << " ("
            << yysym.location << ": ";
        YY_USE (yykind);
        yyo << ')';
      }
  }
#endif

  void
   Parser ::yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym)
  {
    if (m)
      YY_SYMBOL_PRINT (m, sym);
    yystack_.push (YY_MOVE (sym));
  }

  void
   Parser ::yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym)
  {
#if 201103L <= YY_CPLUSPLUS
    yypush_ (m, stack_symbol_type (s, std::move (sym)));
#else
    stack_symbol_type ss (s, sym);
    yypush_ (m, ss);
#endif
  }

  void
   Parser ::yypop_ (int n) YY_NOEXCEPT
  {
    yystack_.pop (n);
  }

#if YYDEBUG
  std::ostream&
   Parser ::debug_stream () const
  {
    return *yycdebug_;
  }

  void
   Parser ::set_debug_stream (std::ostream& o)
  {
    yycdebug_ = &o;
  }


   Parser ::debug_level_type
   Parser ::debug_level () const
  {
    return yydebug_;
  }

  void
   Parser ::set_debug_level (debug_level_type l)
  {
    yydebug_ = l;
  }
#endif // YYDEBUG

   Parser ::state_type
   Parser ::yy_lr_goto_state_ (state_type yystate, int yysym)
  {
    int yyr = yypgoto_[yysym - YYNTOKENS] + yystate;
    if (0 <= yyr && yyr <= yylast_ && yycheck_[yyr] == yystate)
      return yytable_[yyr];
    else
      return yydefgoto_[yysym - YYNTOKENS];
  }

  bool
   Parser ::yy_pact_value_is_default_ (int yyvalue) YY_NOEXCEPT
  {
    return yyvalue == yypact_ninf_;
  }

  bool
   Parser ::yy_table_value_is_error_ (int yyvalue) YY_NOEXCEPT
  {
    return yyvalue == yytable_ninf_;
  }

  int
   Parser ::operator() ()
  {
    return parse ();
  }

  int
   Parser ::parse ()
  {
    int yyn;
    /// Length of the RHS of the rule being reduced.
    int yylen = 0;

    // Error handling.
    int yynerrs_ = 0;
    int yyerrstatus_ = 0;

    /// The lookahead symbol.
    symbol_type yyla;

    /// The locations where the error started and ended.
    stack_symbol_type yyerror_range[3];

    /// The return value of parse ().
    int yyresult;

#if YY_EXCEPTIONS
    try
#endif // YY_EXCEPTIONS
      {
    YYCDEBUG << "Starting parse\n";


    /* Initialize the stack.  The initial state will be set in
       yynewstate, since the latter expects the semantical and the
       location values to have been already stored, initialize these
       stacks with a primary value.  */
    yystack_.clear ();
    yypush_ (YY_NULLPTR, 0, YY_MOVE (yyla));

  /*-----------------------------------------------.
  | yynewstate -- push a new symbol on the stack.  |
  `-----------------------------------------------*/
  yynewstate:
    YYCDEBUG << "Entering state " << int (yystack_[0].state) << '\n';
    YY_STACK_PRINT ();

    // Accept?
    if (yystack_[0].state == yyfinal_)
      YYACCEPT;

    goto yybackup;


  /*-----------.
  | yybackup.  |
  `-----------*/
  yybackup:
    // Try to take a decision without lookahead.
    yyn = yypact_[+yystack_[0].state];
    if (yy_pact_value_is_default_ (yyn))
      goto yydefault;

    // Read a lookahead token.
    if (yyla.empty ())
      {
        YYCDEBUG << "Reading a token\n";
#if YY_EXCEPTIONS
        try
#endif // YY_EXCEPTIONS
          {
            yyla.kind_ = yytranslate_ (yylex (&yyla.value, &yyla.location));
          }
#if YY_EXCEPTIONS
        catch (const syntax_error& yyexc)
          {
            YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
            error (yyexc);
            goto yyerrlab1;
          }
#endif // YY_EXCEPTIONS
      }
    YY_SYMBOL_PRINT ("Next token is", yyla);

    if (yyla.kind () == symbol_kind::S_YYerror)
    {
      // The scanner already issued an error message, process directly
      // to error recovery.  But do not keep the error token as
      // lookahead, it is too special and may lead us to an endless
      // loop in error recovery. */
      yyla.kind_ = symbol_kind::S_YYUNDEF;
      goto yyerrlab1;
    }

    /* If the proper action on seeing token YYLA.TYPE is to reduce or
       to detect an error, take that action.  */
    yyn += yyla.kind ();
    if (yyn < 0 || yylast_ < yyn || yycheck_[yyn] != yyla.kind ())
      {
        goto yydefault;
      }

    // Reduce or error.
    yyn = yytable_[yyn];
    if (yyn <= 0)
      {
        if (yy_table_value_is_error_ (yyn))
          goto yyerrlab;
        yyn = -yyn;
        goto yyreduce;
      }

    // Count tokens shifted since error; after three, turn off error status.
    if (yyerrstatus_)
      --yyerrstatus_;

    // Shift the lookahead token.
    yypush_ ("Shifting", state_type (yyn), YY_MOVE (yyla));
    goto yynewstate;


  /*-----------------------------------------------------------.
  | yydefault -- do the default action for the current state.  |
  `-----------------------------------------------------------*/
  yydefault:
    yyn = yydefact_[+yystack_[0].state];
    if (yyn == 0)
      goto yyerrlab;
    goto yyreduce;


  /*-----------------------------.
  | yyreduce -- do a reduction.  |
  `-----------------------------*/
  yyreduce:
    yylen = yyr2_[yyn];
    {
      stack_symbol_type yylhs;
      yylhs.state = yy_lr_goto_state_ (yystack_[yylen].state, yyr1_[yyn]);
      /* Variants are always initialized to an empty instance of the
         correct type. The default '$$ = $1' action is NOT applied
         when using variants.  */
      switch (yyr1_[yyn])
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        yylhs.value.emplace< TitreInfo > ();
        break;

      case symbol_kind::S_ENTIER: // ENTIER
        yylhs.value.emplace< int > ();
        break;

      case symbol_kind::S_attributs: // attributs
        yylhs.value.emplace< std::map<std::string, std::map<std::string, std::string>> > ();
        break;

      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        yylhs.value.emplace< std::map<std::string, std::string> > ();
        break;

      case symbol_kind::S_PARAGRAPHE: // PARAGRAPHE
      case symbol_kind::S_IMAGE: // IMAGE
      case symbol_kind::S_DEFINE: // DEFINE
      case symbol_kind::S_TITREPAGE: // TITREPAGE
      case symbol_kind::S_STYLE: // STYLE
      case symbol_kind::S_ATTRIBUT: // ATTRIBUT
      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_EGAL: // EGAL
      case symbol_kind::S_CROCHET_FERMANT: // CROCHET_FERMANT
      case symbol_kind::S_CROCHET_OUVRANT: // CROCHET_OUVRANT
      case symbol_kind::S_DEUX_POINTS: // DEUX_POINTS
      case symbol_kind::S_VIRGULE: // VIRGULE
      case symbol_kind::S_POINT_VIRGULE: // POINT_VIRGULE
      case symbol_kind::S_PARENTHESE_OUVRANTE: // PARENTHESE_OUVRANTE
      case symbol_kind::S_PARENTHESE_FERMANTE: // PARENTHESE_FERMANTE
      case symbol_kind::S_ACCOLADE_OUVRANTE: // ACCOLADE_OUVRANTE
      case symbol_kind::S_ACCOLADE_FERMANTE: // ACCOLADE_FERMANTE
      case symbol_kind::S_LARGEUR: // LARGEUR
      case symbol_kind::S_HAUTEUR: // HAUTEUR
      case symbol_kind::S_COULEURTEXTE: // COULEURTEXTE
      case symbol_kind::S_COULEURFOND: // COULEURFOND
      case symbol_kind::S_OPACITE: // OPACITE
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        yylhs.value.emplace< std::string* > ();
        break;

      case symbol_kind::S_element: // element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        yylhs.value.emplace< std::variant<int, std::string, std::unique_ptr<Bloc>> > ();
        break;

      default:
        break;
    }


      // Default location.
      {
        stack_type::slice range (yystack_, yylen);
        YYLLOC_DEFAULT (yylhs.location, range, yylen);
        yyerror_range[1].location = yylhs.location;
      }

      // Perform the reduction.
      YY_REDUCE_PRINT (yyn);
#if YY_EXCEPTIONS
      try
#endif // YY_EXCEPTIONS
        {
          switch (yyn)
            {
  case 9: // element: titre
#line 97 "parser/parser.yy"
    { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > (); }
#line 1116 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 10: // element: sous_titre
#line 98 "parser/parser.yy"
      { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > (); }
#line 1122 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 11: // element: paragraphe
#line 99 "parser/parser.yy"
      { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > (); }
#line 1128 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 12: // element: image
#line 100 "parser/parser.yy"
      { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > (); }
#line 1134 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 13: // element: titrepage
#line 101 "parser/parser.yy"
      { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > (); }
#line 1140 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 14: // element: variable
#line 102 "parser/parser.yy"
      { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > (); }
#line 1146 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 15: // titre: TITRE attributs CHAINE
#line 106 "parser/parser.yy"
                           { 
        yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = std::make_unique<Titre>(yystack_[1].value.as < std::map<std::string, std::map<std::string, std::string>> > (), yystack_[0].value.as < std::string* > ().texte, 1);
        doc->addBloc(std::move(yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
    }
#line 1155 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 16: // titre: TITRE CHAINE
#line 110 "parser/parser.yy"
                   { 
        yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = std::make_unique<Titre>(std::map<std::string, std::string>(), yystack_[0].value.as < std::string* > ().texte, 1);
        doc->addBloc(std::move(yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
    }
#line 1164 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 17: // sous_titre: SOUS_TITRE attributs CHAINE
#line 117 "parser/parser.yy"
                                { 
        yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = std::make_unique<Titre>(yystack_[1].value.as < std::map<std::string, std::map<std::string, std::string>> > (), yystack_[0].value.as < std::string* > ().texte, yystack_[0].value.as < std::string* > ().niveau);
        doc->addBloc(std::move(yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
    }
#line 1173 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 18: // sous_titre: SOUS_TITRE CHAINE
#line 121 "parser/parser.yy"
                        { 
        yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = std::make_unique<Titre>(std::map<std::string, std::string>(), yystack_[0].value.as < std::string* > ().texte, yystack_[0].value.as < std::string* > ().niveau);
        doc->addBloc(std::move(yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
    }
#line 1182 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 19: // paragraphe: PARAGRAPHE attributs CHAINE
#line 128 "parser/parser.yy"
                                { 
        yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = std::make_unique<Paragraphe>(yystack_[1].value.as < std::map<std::string, std::map<std::string, std::string>> > (), yystack_[0].value.as < std::string* > ());
        doc->addBloc(std::move(yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
    }
#line 1191 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 20: // paragraphe: PARAGRAPHE CHAINE
#line 132 "parser/parser.yy"
                        { 
        yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = std::make_unique<Paragraphe>(std::map<std::string, std::string>(), yystack_[0].value.as < std::string* > ());
        doc->addBloc(std::move(yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
    }
#line 1200 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 21: // image: IMAGE CHAINE
#line 139 "parser/parser.yy"
                 { 
        doc->addBloc(std::move(std::make_unique<Image>(yystack_[0].value.as < std::string* > ())));
    }
#line 1208 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 22: // attributs: CROCHET_OUVRANT liste_attributs CROCHET_FERMANT
#line 145 "parser/parser.yy"
                                                    { 
        yylhs.value.as < std::map<std::string, std::map<std::string, std::string>> > () = yystack_[1].value.as < std::map<std::string, std::string> > ();
    }
#line 1216 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 23: // liste_attributs: attribut
#line 151 "parser/parser.yy"
             {
        yylhs.value.as < std::map<std::string, std::string> > () = yystack_[0].value.as < std::map<std::string, std::string> > (); 
    }
#line 1224 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 24: // liste_attributs: attribut VIRGULE liste_attributs
#line 154 "parser/parser.yy"
                                       {
        yylhs.value.as < std::map<std::string, std::string> > () = yystack_[2].value.as < std::map<std::string, std::string> > ();
        yylhs.value.as < std::map<std::string, std::string> > ().insert(yystack_[0].value.as < std::map<std::string, std::string> > ().begin(), yystack_[0].value.as < std::map<std::string, std::string> > ().end());
    }
#line 1233 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 25: // liste_attributs: attribut NEWLINE liste_attributs
#line 158 "parser/parser.yy"
                                       {
        yylhs.value.as < std::map<std::string, std::string> > () = yystack_[2].value.as < std::map<std::string, std::string> > ();
        yylhs.value.as < std::map<std::string, std::string> > ().insert(yystack_[0].value.as < std::map<std::string, std::string> > ().begin(), yystack_[0].value.as < std::map<std::string, std::string> > ().end());
    }
#line 1242 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 26: // attribut: nomattribut DEUX_POINTS valeur
#line 165 "parser/parser.yy"
                                   { 
         yylhs.value.as < std::map<std::string, std::string> > () = std::map<std::string, std::string>{{ yystack_[2].value.as < std::string* > (), yystack_[0].value.as < std::string* > () }};
    }
#line 1250 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 27: // nomattribut: LARGEUR
#line 171 "parser/parser.yy"
            { yylhs.value.as < std::string* > () = "width"; }
#line 1256 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 28: // nomattribut: HAUTEUR
#line 172 "parser/parser.yy"
              { yylhs.value.as < std::string* > () = "height"; }
#line 1262 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 29: // nomattribut: COULEURTEXTE
#line 173 "parser/parser.yy"
                   { yylhs.value.as < std::string* > () = "color"; }
#line 1268 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 30: // nomattribut: COULEURFOND
#line 174 "parser/parser.yy"
                  { yylhs.value.as < std::string* > () = "background-color"; }
#line 1274 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 31: // nomattribut: OPACITE
#line 175 "parser/parser.yy"
              { yylhs.value.as < std::string* > () = "opacity"; }
#line 1280 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 32: // valeur: ENTIER
#line 179 "parser/parser.yy"
           { yylhs.value.as < std::string* > () = yystack_[0].value.as < int > (); }
#line 1286 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 33: // valeur: HEX_COULEUR
#line 180 "parser/parser.yy"
                  { yylhs.value.as < std::string* > () = *yystack_[0].value.as < std::string* > (); }
#line 1292 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 34: // valeur: RGB_COULEUR
#line 181 "parser/parser.yy"
                  { yylhs.value.as < std::string* > () = *yystack_[0].value.as < std::string* > (); }
#line 1298 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 35: // valeur: CHAINE
#line 182 "parser/parser.yy"
             { yylhs.value.as < std::string* > () = *yystack_[0].value.as < std::string* > (); }
#line 1304 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 36: // define: DEFINE PARENTHESE_OUVRANTE PROPRIETE PARENTHESE_FERMANTE ACCOLADE_OUVRANTE valeur ACCOLADE_FERMANTE
#line 187 "parser/parser.yy"
    { 
        doc->setPropriete(*yystack_[4].value.as < std::string* > (), *yystack_[1].value.as < std::string* > ());
    }
#line 1312 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 37: // titrepage: TITREPAGE CHAINE
#line 193 "parser/parser.yy"
                     { 
        yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = std::make_unique<TitrePage>(yystack_[0].value.as < std::string* > ());
        doc->addBloc(std::move(yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
    }
#line 1321 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 38: // variable: IDENTIFIANT EGAL valeurvar
#line 200 "parser/parser.yy"
                               { 
        if (std::holds_alternative<std::unique_ptr<Bloc>>(yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ())) {
            doc->setVariable(yystack_[2].value.as < std::string* > (), std::move(std::get<std::unique_ptr<Bloc>>(yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ())));
        } else if (std::holds_alternative<int>(yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ())) {
            doc->setVariable(yystack_[2].value.as < std::string* > (), std::get<int>(yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
        } else if (std::holds_alternative<std::string>(yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ())) {
            doc->setVariable(yystack_[2].value.as < std::string* > (), std::get<std::string>(yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()));
        }
    }
#line 1335 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 39: // valeurvar: ENTIER
#line 212 "parser/parser.yy"
           { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = yystack_[0].value.as < int > (); }
#line 1341 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 40: // valeurvar: HEX_COULEUR
#line 213 "parser/parser.yy"
                  { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = *yystack_[0].value.as < std::string* > (); }
#line 1347 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 41: // valeurvar: RGB_COULEUR
#line 214 "parser/parser.yy"
                  { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = *yystack_[0].value.as < std::string* > (); }
#line 1353 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 42: // valeurvar: element
#line 215 "parser/parser.yy"
              { yylhs.value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > () = std::move(yystack_[0].value.as < std::variant<int, std::string, std::unique_ptr<Bloc>> > ()); }
#line 1359 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 43: // style: STYLE PARENTHESE_OUVRANTE IDENTIFIANT PARENTHESE_FERMANTE ACCOLADE_OUVRANTE attributs ACCOLADE_FERMANTE
#line 220 "parser/parser.yy"
    { 
        doc->setStyle(yystack_[4].value.as < std::string* > (), yystack_[1].value.as < std::map<std::string, std::map<std::string, std::string>> > ());
    }
#line 1367 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;


#line 1371 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"

            default:
              break;
            }
        }
#if YY_EXCEPTIONS
      catch (const syntax_error& yyexc)
        {
          YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
          error (yyexc);
          YYERROR;
        }
#endif // YY_EXCEPTIONS
      YY_SYMBOL_PRINT ("-> $$ =", yylhs);
      yypop_ (yylen);
      yylen = 0;

      // Shift the result of the reduction.
      yypush_ (YY_NULLPTR, YY_MOVE (yylhs));
    }
    goto yynewstate;


  /*--------------------------------------.
  | yyerrlab -- here on detecting error.  |
  `--------------------------------------*/
  yyerrlab:
    // If not already recovering from an error, report this error.
    if (!yyerrstatus_)
      {
        ++yynerrs_;
        std::string msg = YY_("syntax error");
        error (yyla.location, YY_MOVE (msg));
      }


    yyerror_range[1].location = yyla.location;
    if (yyerrstatus_ == 3)
      {
        /* If just tried and failed to reuse lookahead token after an
           error, discard it.  */

        // Return failure if at end of input.
        if (yyla.kind () == symbol_kind::S_YYEOF)
          YYABORT;
        else if (!yyla.empty ())
          {
            yy_destroy_ ("Error: discarding", yyla);
            yyla.clear ();
          }
      }

    // Else will try to reuse lookahead token after shifting the error token.
    goto yyerrlab1;


  /*---------------------------------------------------.
  | yyerrorlab -- error raised explicitly by YYERROR.  |
  `---------------------------------------------------*/
  yyerrorlab:
    /* Pacify compilers when the user code never invokes YYERROR and
       the label yyerrorlab therefore never appears in user code.  */
    if (false)
      YYERROR;

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYERROR.  */
    yypop_ (yylen);
    yylen = 0;
    YY_STACK_PRINT ();
    goto yyerrlab1;


  /*-------------------------------------------------------------.
  | yyerrlab1 -- common code for both syntax error and YYERROR.  |
  `-------------------------------------------------------------*/
  yyerrlab1:
    yyerrstatus_ = 3;   // Each real token shifted decrements this.
    // Pop stack until we find a state that shifts the error token.
    for (;;)
      {
        yyn = yypact_[+yystack_[0].state];
        if (!yy_pact_value_is_default_ (yyn))
          {
            yyn += symbol_kind::S_YYerror;
            if (0 <= yyn && yyn <= yylast_
                && yycheck_[yyn] == symbol_kind::S_YYerror)
              {
                yyn = yytable_[yyn];
                if (0 < yyn)
                  break;
              }
          }

        // Pop the current state because it cannot handle the error token.
        if (yystack_.size () == 1)
          YYABORT;

        yyerror_range[1].location = yystack_[0].location;
        yy_destroy_ ("Error: popping", yystack_[0]);
        yypop_ ();
        YY_STACK_PRINT ();
      }
    {
      stack_symbol_type error_token;

      yyerror_range[2].location = yyla.location;
      YYLLOC_DEFAULT (error_token.location, yyerror_range, 2);

      // Shift the error token.
      error_token.state = state_type (yyn);
      yypush_ ("Shifting", YY_MOVE (error_token));
    }
    goto yynewstate;


  /*-------------------------------------.
  | yyacceptlab -- YYACCEPT comes here.  |
  `-------------------------------------*/
  yyacceptlab:
    yyresult = 0;
    goto yyreturn;


  /*-----------------------------------.
  | yyabortlab -- YYABORT comes here.  |
  `-----------------------------------*/
  yyabortlab:
    yyresult = 1;
    goto yyreturn;


  /*-----------------------------------------------------.
  | yyreturn -- parsing is finished, return the result.  |
  `-----------------------------------------------------*/
  yyreturn:
    if (!yyla.empty ())
      yy_destroy_ ("Cleanup: discarding lookahead", yyla);

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    yypop_ (yylen);
    YY_STACK_PRINT ();
    while (1 < yystack_.size ())
      {
        yy_destroy_ ("Cleanup: popping", yystack_[0]);
        yypop_ ();
      }

    return yyresult;
  }
#if YY_EXCEPTIONS
    catch (...)
      {
        YYCDEBUG << "Exception caught: cleaning lookahead and stack\n";
        // Do not try to display the values of the reclaimed symbols,
        // as their printers might throw an exception.
        if (!yyla.empty ())
          yy_destroy_ (YY_NULLPTR, yyla);

        while (1 < yystack_.size ())
          {
            yy_destroy_ (YY_NULLPTR, yystack_[0]);
            yypop_ ();
          }
        throw;
      }
#endif // YY_EXCEPTIONS
  }

  void
   Parser ::error (const syntax_error& yyexc)
  {
    error (yyexc.location, yyexc.what ());
  }

#if YYDEBUG || 0
  const char *
   Parser ::symbol_name (symbol_kind_type yysymbol)
  {
    return yytname_[yysymbol];
  }
#endif // #if YYDEBUG || 0









  const signed char  Parser ::yypact_ninf_ = -45;

  const signed char  Parser ::yytable_ninf_ = -1;

  const signed char
   Parser ::yypact_[] =
  {
      19,   -15,    -9,    40,     6,    19,   -45,   -45,    29,    24,
     -45,   -17,     3,     5,    23,    25,    21,   -45,     6,   -45,
     -45,   -45,   -45,   -45,   -45,   -45,    18,    20,   -45,    -2,
      27,   -45,    31,   -45,    32,   -45,   -45,     0,   -45,    22,
      26,   -45,   -45,   -45,   -45,   -45,    30,    -1,    33,   -45,
     -45,   -45,   -45,   -45,   -45,   -45,   -45,    17,    35,   -45,
      -2,    -2,    17,   -45,   -45,   -45,   -45,    34,    36,   -45,
     -45,   -45,   -45,   -45
  };

  const signed char
   Parser ::yydefact_[] =
  {
       6,     0,     0,     0,     4,     6,     7,     8,     0,     0,
       1,     0,     0,     0,     0,     0,     0,     2,     4,     9,
      10,    11,    12,    13,    14,     5,     0,     0,    16,     0,
       0,    18,     0,    20,     0,    21,    37,     0,     3,     0,
       0,    27,    28,    29,    30,    31,     0,    23,     0,    15,
      17,    19,    39,    40,    41,    42,    38,     0,     0,    22,
       0,     0,     0,    32,    35,    33,    34,     0,     0,    25,
      24,    26,    36,    43
  };

  const signed char
   Parser ::yypgoto_[] =
  {
     -45,   -45,    37,    44,   -45,    28,   -45,   -45,   -45,   -45,
     -12,   -44,   -45,   -45,    -6,   -45,   -45,   -45,   -45,   -45
  };

  const signed char
   Parser ::yydefgoto_[] =
  {
       0,     3,    17,     4,     5,    18,    19,    20,    21,    22,
      30,    46,    47,    48,    67,     6,    23,    24,    56,     7
  };

  const signed char
   Parser ::yytable_[] =
  {
      32,    34,    60,    28,    11,    12,    13,    14,    29,    15,
      11,    12,    13,    14,     8,    15,    69,    70,    16,    52,
       9,    53,    54,    31,    16,    33,    61,     1,    29,     2,
      29,    41,    42,    43,    44,    45,    63,    64,    65,    66,
      10,    26,    27,    35,    37,    36,    68,    49,    39,    25,
      40,    50,    51,    57,    59,    38,    71,    58,     0,    62,
      29,     0,     0,     0,     0,    55,    72,     0,    73
  };

  const signed char
   Parser ::yycheck_[] =
  {
      12,    13,     3,    20,     4,     5,     6,     7,    25,     9,
       4,     5,     6,     7,    29,     9,    60,    61,    18,    19,
      29,    21,    22,    20,    18,    20,    27,     8,    25,    10,
      25,    33,    34,    35,    36,    37,    19,    20,    21,    22,
       0,    12,    18,    20,    23,    20,    58,    20,    30,     5,
      30,    20,    20,    31,    24,    18,    62,    31,    -1,    26,
      25,    -1,    -1,    -1,    -1,    37,    32,    -1,    32
  };

  const signed char
   Parser ::yystos_[] =
  {
       0,     8,    10,    39,    41,    42,    53,    57,    29,    29,
       0,     4,     5,     6,     7,     9,    18,    40,    43,    44,
      45,    46,    47,    54,    55,    41,    12,    18,    20,    25,
      48,    20,    48,    20,    48,    20,    20,    23,    40,    30,
      30,    33,    34,    35,    36,    37,    49,    50,    51,    20,
      20,    20,    19,    21,    22,    43,    56,    31,    31,    24,
       3,    27,    26,    19,    20,    21,    22,    52,    48,    49,
      49,    52,    32,    32
  };

  const signed char
   Parser ::yyr1_[] =
  {
       0,    38,    39,    40,    40,    41,    41,    42,    42,    43,
      43,    43,    43,    43,    43,    44,    44,    45,    45,    46,
      46,    47,    48,    49,    49,    49,    50,    51,    51,    51,
      51,    51,    52,    52,    52,    52,    53,    54,    55,    56,
      56,    56,    56,    57
  };

  const signed char
   Parser ::yyr2_[] =
  {
       0,     2,     2,     2,     0,     2,     0,     1,     1,     1,
       1,     1,     1,     1,     1,     3,     2,     3,     2,     3,
       2,     2,     3,     1,     3,     3,     3,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     7,     2,     3,     1,
       1,     1,     1,     7
  };


#if YYDEBUG
  // YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
  // First, the terminals, then, starting at \a YYNTOKENS, nonterminals.
  const char*
  const  Parser ::yytname_[] =
  {
  "\"end of file\"", "error", "\"invalid token\"", "NEWLINE", "TITRE",
  "SOUS_TITRE", "PARAGRAPHE", "IMAGE", "DEFINE", "TITREPAGE", "STYLE",
  "ATTRIBUT", "PROPRIETE", "SI", "SINON", "FINSI", "POUR", "FINI",
  "IDENTIFIANT", "ENTIER", "CHAINE", "HEX_COULEUR", "RGB_COULEUR", "EGAL",
  "CROCHET_FERMANT", "CROCHET_OUVRANT", "DEUX_POINTS", "VIRGULE",
  "POINT_VIRGULE", "PARENTHESE_OUVRANTE", "PARENTHESE_FERMANTE",
  "ACCOLADE_OUVRANTE", "ACCOLADE_FERMANTE", "LARGEUR", "HAUTEUR",
  "COULEURTEXTE", "COULEURFOND", "OPACITE", "$accept", "programme",
  "elements", "declarations", "declaration", "element", "titre",
  "sous_titre", "paragraphe", "image", "attributs", "liste_attributs",
  "attribut", "nomattribut", "valeur", "define", "titrepage", "variable",
  "valeurvar", "style", YY_NULLPTR
  };
#endif


#if YYDEBUG
  const unsigned char
   Parser ::yyrline_[] =
  {
       0,    78,    78,    82,    83,    87,    88,    92,    93,    97,
      98,    99,   100,   101,   102,   106,   110,   117,   121,   128,
     132,   139,   145,   151,   154,   158,   165,   171,   172,   173,
     174,   175,   179,   180,   181,   182,   186,   193,   200,   212,
     213,   214,   215,   219
  };

  void
   Parser ::yy_stack_print_ () const
  {
    *yycdebug_ << "Stack now";
    for (stack_type::const_iterator
           i = yystack_.begin (),
           i_end = yystack_.end ();
         i != i_end; ++i)
      *yycdebug_ << ' ' << int (i->state);
    *yycdebug_ << '\n';
  }

  void
   Parser ::yy_reduce_print_ (int yyrule) const
  {
    int yylno = yyrline_[yyrule];
    int yynrhs = yyr2_[yyrule];
    // Print the symbols being reduced, and their result.
    *yycdebug_ << "Reducing stack by rule " << yyrule - 1
               << " (line " << yylno << "):\n";
    // The symbols being reduced.
    for (int yyi = 0; yyi < yynrhs; yyi++)
      YY_SYMBOL_PRINT ("   $" << yyi + 1 << " =",
                       yystack_[(yynrhs) - (yyi + 1)]);
  }
#endif // YYDEBUG

   Parser ::symbol_kind_type
   Parser ::yytranslate_ (int t) YY_NOEXCEPT
  {
    // YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to
    // TOKEN-NUM as returned by yylex.
    static
    const signed char
    translate_table[] =
    {
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37
    };
    // Last valid token kind.
    const int code_max = 292;

    if (t <= 0)
      return symbol_kind::S_YYEOF;
    else if (t <= code_max)
      return static_cast <symbol_kind_type> (translate_table[t]);
    else
      return symbol_kind::S_YYUNDEF;
  }

} // yy
#line 1778 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"

#line 225 "parser/parser.yy"


void yy::Parser::error( const location_type &l, const std::string & err_msg) {
    std::cerr << "Erreur : " << l << ", " << err_msg << std::endl;
}
