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
#line 30 "parser/parser.yy"

    #include <iostream>
    #include <string>
    #include <memory>
    #include <map>
    #include <variant>
    
    #include "scanner.hh"
    #include "driver.hh"


    #undef  yylex
    #define yylex scanner.yylex

#line 61 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"


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
#line 153 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"

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
      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
        value.copy< Bloc* > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.copy< TitreInfo > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        value.copy< int > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.copy< std::map<std::string, std::string> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_selecteur: // selecteur
        value.copy< std::pair<std::string, int> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.copy< std::string > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.copy< std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > (YY_MOVE (that.value));
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
      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
        value.move< Bloc* > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.move< TitreInfo > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        value.move< int > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.move< std::map<std::string, std::string> > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_selecteur: // selecteur
        value.move< std::pair<std::string, int> > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.move< std::string > (YY_MOVE (s.value));
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.move< std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > (YY_MOVE (s.value));
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
      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
        value.YY_MOVE_OR_COPY< Bloc* > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.YY_MOVE_OR_COPY< TitreInfo > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        value.YY_MOVE_OR_COPY< int > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.YY_MOVE_OR_COPY< std::map<std::string, std::string> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_selecteur: // selecteur
        value.YY_MOVE_OR_COPY< std::pair<std::string, int> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.YY_MOVE_OR_COPY< std::string > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.YY_MOVE_OR_COPY< std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > (YY_MOVE (that.value));
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
      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
        value.move< Bloc* > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.move< TitreInfo > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        value.move< int > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.move< std::map<std::string, std::string> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_selecteur: // selecteur
        value.move< std::pair<std::string, int> > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.move< std::string > (YY_MOVE (that.value));
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.move< std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > (YY_MOVE (that.value));
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
      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
        value.copy< Bloc* > (that.value);
        break;

      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.copy< TitreInfo > (that.value);
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        value.copy< int > (that.value);
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.copy< std::map<std::string, std::string> > (that.value);
        break;

      case symbol_kind::S_selecteur: // selecteur
        value.copy< std::pair<std::string, int> > (that.value);
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.copy< std::string > (that.value);
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.copy< std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > (that.value);
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
      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
        value.move< Bloc* > (that.value);
        break;

      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.move< TitreInfo > (that.value);
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        value.move< int > (that.value);
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.move< std::map<std::string, std::string> > (that.value);
        break;

      case symbol_kind::S_selecteur: // selecteur
        value.move< std::pair<std::string, int> > (that.value);
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.move< std::string > (that.value);
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.move< std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > (that.value);
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
      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
        yylhs.value.emplace< Bloc* > ();
        break;

      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        yylhs.value.emplace< TitreInfo > ();
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        yylhs.value.emplace< int > ();
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        yylhs.value.emplace< std::map<std::string, std::string> > ();
        break;

      case symbol_kind::S_selecteur: // selecteur
        yylhs.value.emplace< std::pair<std::string, int> > ();
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_SI: // SI
      case symbol_kind::S_SINON: // SINON
      case symbol_kind::S_FINSI: // FINSI
      case symbol_kind::S_POUR: // POUR
      case symbol_kind::S_FINI: // FINI
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        yylhs.value.emplace< std::string > ();
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        yylhs.value.emplace< std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ();
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
  case 11: // bloc_element: titre
#line 90 "parser/parser.yy"
    { yylhs.value.as < Bloc* > () = yystack_[0].value.as < Bloc* > (); }
#line 1042 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 12: // bloc_element: sous_titre
#line 91 "parser/parser.yy"
      { yylhs.value.as < Bloc* > () = yystack_[0].value.as < Bloc* > (); }
#line 1048 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 13: // bloc_element: paragraphe
#line 92 "parser/parser.yy"
      { yylhs.value.as < Bloc* > () = yystack_[0].value.as < Bloc* > (); }
#line 1054 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 14: // bloc_element: image
#line 93 "parser/parser.yy"
      { yylhs.value.as < Bloc* > () = yystack_[0].value.as < Bloc* > (); }
#line 1060 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 15: // titre: TITRE attributs CHAINE
#line 97 "parser/parser.yy"
                           { 
        yylhs.value.as < Bloc* > () = new Titre(yystack_[1].value.as < std::map<std::string, std::string> > (), yystack_[0].value.as < std::string > (), yystack_[2].value.as < TitreInfo > ().niveau);
        doc->addBloc(yylhs.value.as < Bloc* > ());
    }
#line 1069 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 16: // titre: TITRE CHAINE
#line 101 "parser/parser.yy"
                   { 
        yylhs.value.as < Bloc* > () = new Titre(std::map<std::string, std::string>(), yystack_[0].value.as < std::string > (), yystack_[1].value.as < TitreInfo > ().niveau);
        doc->addBloc(yylhs.value.as < Bloc* > ());
    }
#line 1078 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 17: // sous_titre: SOUS_TITRE attributs CHAINE
#line 108 "parser/parser.yy"
                                { 
        yylhs.value.as < Bloc* > () = new Titre(yystack_[1].value.as < std::map<std::string, std::string> > (), yystack_[0].value.as < std::string > (), yystack_[2].value.as < TitreInfo > ().niveau);
        doc->addBloc(yylhs.value.as < Bloc* > ());
    }
#line 1087 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 18: // sous_titre: SOUS_TITRE CHAINE
#line 112 "parser/parser.yy"
                        { 
        yylhs.value.as < Bloc* > () = new Titre(std::map<std::string, std::string>(), yystack_[0].value.as < std::string > (), yystack_[1].value.as < TitreInfo > ().niveau);
        doc->addBloc(yylhs.value.as < Bloc* > ());
    }
#line 1096 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 19: // paragraphe: PARAGRAPHE attributs CHAINE
#line 119 "parser/parser.yy"
                                { 
        yylhs.value.as < Bloc* > () = new Paragraphe(yystack_[1].value.as < std::map<std::string, std::string> > (), yystack_[0].value.as < std::string > ());
        doc->addBloc(yylhs.value.as < Bloc* > ());
    }
#line 1105 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 20: // paragraphe: PARAGRAPHE CHAINE
#line 123 "parser/parser.yy"
                        { 
        yylhs.value.as < Bloc* > () = new Paragraphe(std::map<std::string, std::string>(), yystack_[0].value.as < std::string > ());
        doc->addBloc(yylhs.value.as < Bloc* > ());
    }
#line 1114 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 21: // image: IMAGE CHAINE
#line 130 "parser/parser.yy"
                 { 
        doc->addBloc(new Image(yystack_[0].value.as < std::string > ()));
    }
#line 1122 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 22: // commentaire: COMMENTAIRE
#line 136 "parser/parser.yy"
                { 
        doc->addBloc(new Commentaire(yystack_[0].value.as < std::string > ()));
    }
#line 1130 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 23: // attributs: CROCHET_OUVRANT liste_attributs CROCHET_FERMANT
#line 142 "parser/parser.yy"
                                                    { 
        yylhs.value.as < std::map<std::string, std::string> > () = yystack_[1].value.as < std::map<std::string, std::string> > ();
    }
#line 1138 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 24: // liste_attributs: attribut
#line 148 "parser/parser.yy"
             {
        yylhs.value.as < std::map<std::string, std::string> > () = yystack_[0].value.as < std::map<std::string, std::string> > (); 
    }
#line 1146 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 25: // liste_attributs: attribut VIRGULE liste_attributs
#line 151 "parser/parser.yy"
                                       {
        yylhs.value.as < std::map<std::string, std::string> > () = yystack_[2].value.as < std::map<std::string, std::string> > ();
        yylhs.value.as < std::map<std::string, std::string> > ().insert(yystack_[0].value.as < std::map<std::string, std::string> > ().begin(), yystack_[0].value.as < std::map<std::string, std::string> > ().end());
    }
#line 1155 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 26: // liste_attributs: attribut liste_attributs
#line 155 "parser/parser.yy"
                               {
        yylhs.value.as < std::map<std::string, std::string> > () = yystack_[1].value.as < std::map<std::string, std::string> > ();
        yylhs.value.as < std::map<std::string, std::string> > ().insert(yystack_[0].value.as < std::map<std::string, std::string> > ().begin(), yystack_[0].value.as < std::map<std::string, std::string> > ().end());
    }
#line 1164 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 27: // attribut: nomattribut DEUX_POINTS valeur
#line 162 "parser/parser.yy"
                                   { 
         yylhs.value.as < std::map<std::string, std::string> > () = std::map<std::string, std::string>{{ yystack_[2].value.as < std::string > (), yystack_[0].value.as < std::string > () }};
    }
#line 1172 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 28: // attribut: nomattribut DEUX_POINTS IDENTIFIANT
#line 165 "parser/parser.yy"
                                          {
        std::string val = std::get<std::string>(doc->getVariable(yystack_[0].value.as < std::string > ()));
        if (val != "") {
            yylhs.value.as < std::map<std::string, std::string> > () = std::map<std::string, std::string>{{ yystack_[2].value.as < std::string > (), val }};
        }
    }
#line 1183 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 29: // nomattribut: LARGEUR
#line 174 "parser/parser.yy"
            { yylhs.value.as < std::string > () = "width"; }
#line 1189 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 30: // nomattribut: HAUTEUR
#line 175 "parser/parser.yy"
              { yylhs.value.as < std::string > () = "height"; }
#line 1195 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 31: // nomattribut: COULEURTEXTE
#line 176 "parser/parser.yy"
                   { yylhs.value.as < std::string > () = "color"; }
#line 1201 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 32: // nomattribut: COULEURFOND
#line 177 "parser/parser.yy"
                  { yylhs.value.as < std::string > () = "background-color"; }
#line 1207 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 33: // nomattribut: OPACITE
#line 178 "parser/parser.yy"
              { yylhs.value.as < std::string > () = "opacity"; }
#line 1213 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 34: // valeur: ENTIER
#line 182 "parser/parser.yy"
           { yylhs.value.as < std::string > () = std::to_string(yystack_[0].value.as < int > ()); }
#line 1219 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 35: // valeur: HEX_COULEUR
#line 183 "parser/parser.yy"
                  { yylhs.value.as < std::string > () = yystack_[0].value.as < std::string > (); }
#line 1225 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 36: // valeur: RGB_COULEUR
#line 184 "parser/parser.yy"
                  { yylhs.value.as < std::string > () = yystack_[0].value.as < std::string > (); }
#line 1231 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 37: // valeur: CHAINE
#line 185 "parser/parser.yy"
             { yylhs.value.as < std::string > () = yystack_[0].value.as < std::string > (); }
#line 1237 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 38: // define: DEFINE PARENTHESE_OUVRANTE PROPRIETE PARENTHESE_FERMANTE ACCOLADE_OUVRANTE valeur ACCOLADE_FERMANTE
#line 190 "parser/parser.yy"
    { 
        doc->setPropriete(yystack_[4].value.as < std::string > (), yystack_[1].value.as < std::string > ());
    }
#line 1245 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 39: // titrepage: TITREPAGE CHAINE
#line 196 "parser/parser.yy"
                     { 
        auto bloc = new TitrePage(yystack_[0].value.as < std::string > ());
        doc->addBloc(bloc);
    }
#line 1254 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 40: // variable: IDENTIFIANT EGAL valeurvar
#line 203 "parser/parser.yy"
                               { 
        if (std::holds_alternative<Bloc*>(yystack_[0].value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ())) {
            doc->setVariable(yystack_[2].value.as < std::string > (), std::get<Bloc*>(yystack_[0].value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ()));
        } else if (std::holds_alternative<int>(yystack_[0].value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ())) {
            doc->setVariable(yystack_[2].value.as < std::string > (), std::get<int>(yystack_[0].value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ()));
        } else if (std::holds_alternative<std::string>(yystack_[0].value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ())) {
            doc->setVariable(yystack_[2].value.as < std::string > (), std::get<std::string>(yystack_[0].value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ()));
        } else if (std::holds_alternative<std::map<std::string, std::string>>(yystack_[0].value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ())) {
            doc->setVariable(yystack_[2].value.as < std::string > (), std::get<std::map<std::string, std::string>>(yystack_[0].value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > ()));
        }

    }
#line 1271 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 41: // variable: IDENTIFIANT EGAL selecteur
#line 215 "parser/parser.yy"
                                 { 
        Bloc *b = doc->getNBloc(yystack_[0].value.as < std::pair<std::string, int> > ().first, yystack_[0].value.as < std::pair<std::string, int> > ().second);
        if (b != nullptr) {
            doc->setVariable(yystack_[2].value.as < std::string > (), b);
        }
    }
#line 1282 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 42: // variable: IDENTIFIANT POINT nomattribut EGAL valeur
#line 221 "parser/parser.yy"
                                                {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable(yystack_[4].value.as < std::string > ()));
        if (bloc != nullptr) {
            bloc->setPropriete(yystack_[2].value.as < std::string > (), yystack_[0].value.as < std::string > ());
        }
    }
#line 1293 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 43: // variable: IDENTIFIANT POINT nomattribut EGAL IDENTIFIANT
#line 227 "parser/parser.yy"
                                                     {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable(yystack_[4].value.as < std::string > ()));
        if (bloc != nullptr) {
            std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> prop = doc->getVariable(yystack_[0].value.as < std::string > ());
            if (std::holds_alternative<std::string>(prop)) {
                bloc->setPropriete(yystack_[2].value.as < std::string > (), std::get<std::string>(prop));
            }
        }
    }
#line 1307 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 44: // variable: IDENTIFIANT POINT SELECTSTYLE EGAL IDENTIFIANT
#line 236 "parser/parser.yy"
                                                     {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable(yystack_[4].value.as < std::string > ()));
        if (bloc != nullptr) {
            std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> prop = doc->getVariable(yystack_[0].value.as < std::string > ());
            if (std::holds_alternative<std::map<std::string, std::string>>(prop)) {
                for (auto const& [key, val] : std::get<std::map<std::string, std::string>>(prop)) {
                    bloc->setPropriete(key, val);
                }
            }
        }
    }
#line 1323 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 45: // variable: IDENTIFIANT POINT SELECTSTYLE EGAL attributs
#line 247 "parser/parser.yy"
                                                   {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable(yystack_[4].value.as < std::string > ()));
        if (bloc != nullptr) {
            for (auto const& [key, val] : yystack_[0].value.as < std::map<std::string, std::string> > ()) {
                bloc->setPropriete(key, val);
            }
        }
    }
#line 1336 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 46: // selecteur: PARAGRAPHE index_expression
#line 259 "parser/parser.yy"
                                { yylhs.value.as < std::pair<std::string, int> > () = std::make_pair("p", yystack_[0].value.as < int > ()); }
#line 1342 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 47: // selecteur: TITRE index_expression
#line 260 "parser/parser.yy"
                                  { yylhs.value.as < std::pair<std::string, int> > () = std::make_pair("h", yystack_[0].value.as < int > ()); }
#line 1348 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 48: // selecteur: SOUS_TITRE index_expression
#line 261 "parser/parser.yy"
                                  { yylhs.value.as < std::pair<std::string, int> > () = std::make_pair("h", yystack_[0].value.as < int > ()); }
#line 1354 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 49: // selecteur: IMAGE index_expression
#line 262 "parser/parser.yy"
                                  { yylhs.value.as < std::pair<std::string, int> > () = std::make_pair("img", yystack_[0].value.as < int > ()); }
#line 1360 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 50: // index_expression: CROCHET_OUVRANT expr CROCHET_FERMANT
#line 266 "parser/parser.yy"
                                         { yylhs.value.as < int > () = yystack_[1].value.as < int > (); }
#line 1366 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 51: // expr: expr PLUS terme
#line 270 "parser/parser.yy"
                    { yylhs.value.as < int > () = yystack_[2].value.as < int > () + yystack_[0].value.as < int > (); }
#line 1372 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 52: // expr: expr MOINS terme
#line 271 "parser/parser.yy"
                       { yylhs.value.as < int > () = yystack_[2].value.as < int > () - yystack_[0].value.as < int > (); }
#line 1378 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 53: // expr: terme
#line 272 "parser/parser.yy"
            { yylhs.value.as < int > () = yystack_[0].value.as < int > (); }
#line 1384 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 54: // terme: terme MULT facteur
#line 276 "parser/parser.yy"
                       { yylhs.value.as < int > () = yystack_[2].value.as < int > () * yystack_[0].value.as < int > (); }
#line 1390 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 55: // terme: terme DIV facteur
#line 277 "parser/parser.yy"
                        { yylhs.value.as < int > () = yystack_[2].value.as < int > () / yystack_[0].value.as < int > (); }
#line 1396 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 56: // terme: facteur
#line 278 "parser/parser.yy"
              { yylhs.value.as < int > () = yystack_[0].value.as < int > (); }
#line 1402 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 57: // facteur: ENTIER
#line 282 "parser/parser.yy"
           { yylhs.value.as < int > () = yystack_[0].value.as < int > (); }
#line 1408 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 58: // facteur: IDENTIFIANT
#line 283 "parser/parser.yy"
                  {
        auto val = doc->getVariable(yystack_[0].value.as < std::string > ());
        if(std::holds_alternative<int>(val)) {
            yylhs.value.as < int > () = std::get<int>(val);
        } else {
            std::cerr << "Erreur: la variable " << yystack_[0].value.as < std::string > () << " n'est pas un entier" << std::endl;
            yylhs.value.as < int > () = -1;
        }
    }
#line 1422 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 59: // valeurvar: ENTIER
#line 294 "parser/parser.yy"
           { yylhs.value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > () = yystack_[0].value.as < int > (); }
#line 1428 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 60: // valeurvar: HEX_COULEUR
#line 295 "parser/parser.yy"
                  { yylhs.value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > () = yystack_[0].value.as < std::string > (); }
#line 1434 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 61: // valeurvar: RGB_COULEUR
#line 296 "parser/parser.yy"
                  { yylhs.value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > () = yystack_[0].value.as < std::string > (); }
#line 1440 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 62: // valeurvar: bloc_element
#line 297 "parser/parser.yy"
                   { yylhs.value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > () = std::variant<int, std::string, Bloc*, std::map<std::string, std::string>>(yystack_[0].value.as < Bloc* > ()); }
#line 1446 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 63: // valeurvar: attributs
#line 298 "parser/parser.yy"
                { yylhs.value.as < std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> > () = yystack_[0].value.as < std::map<std::string, std::string> > (); }
#line 1452 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;

  case 64: // style: STYLE PARENTHESE_OUVRANTE BLOCS PARENTHESE_FERMANTE CROCHET_OUVRANT liste_attributs CROCHET_FERMANT
#line 303 "parser/parser.yy"
    { 
        doc->setStyle(yystack_[4].value.as < std::string > (), yystack_[1].value.as < std::map<std::string, std::string> > ());
    }
#line 1460 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"
    break;


#line 1464 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"

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
        context yyctx (*this, yyla);
        std::string msg = yysyntax_error_ (yyctx);
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

  /* Return YYSTR after stripping away unnecessary quotes and
     backslashes, so that it's suitable for yyerror.  The heuristic is
     that double-quoting is unnecessary unless the string contains an
     apostrophe, a comma, or backslash (other than backslash-backslash).
     YYSTR is taken from yytname.  */
  std::string
   Parser ::yytnamerr_ (const char *yystr)
  {
    if (*yystr == '"')
      {
        std::string yyr;
        char const *yyp = yystr;

        for (;;)
          switch (*++yyp)
            {
            case '\'':
            case ',':
              goto do_not_strip_quotes;

            case '\\':
              if (*++yyp != '\\')
                goto do_not_strip_quotes;
              else
                goto append;

            append:
            default:
              yyr += *yyp;
              break;

            case '"':
              return yyr;
            }
      do_not_strip_quotes: ;
      }

    return yystr;
  }

  std::string
   Parser ::symbol_name (symbol_kind_type yysymbol)
  {
    return yytnamerr_ (yytname_[yysymbol]);
  }



  //  Parser ::context.
   Parser ::context::context (const  Parser & yyparser, const symbol_type& yyla)
    : yyparser_ (yyparser)
    , yyla_ (yyla)
  {}

  int
   Parser ::context::expected_tokens (symbol_kind_type yyarg[], int yyargn) const
  {
    // Actual number of expected tokens
    int yycount = 0;

    const int yyn = yypact_[+yyparser_.yystack_[0].state];
    if (!yy_pact_value_is_default_ (yyn))
      {
        /* Start YYX at -YYN if negative to avoid negative indexes in
           YYCHECK.  In other words, skip the first -YYN actions for
           this state because they are default actions.  */
        const int yyxbegin = yyn < 0 ? -yyn : 0;
        // Stay within bounds of both yycheck and yytname.
        const int yychecklim = yylast_ - yyn + 1;
        const int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
        for (int yyx = yyxbegin; yyx < yyxend; ++yyx)
          if (yycheck_[yyx + yyn] == yyx && yyx != symbol_kind::S_YYerror
              && !yy_table_value_is_error_ (yytable_[yyx + yyn]))
            {
              if (!yyarg)
                ++yycount;
              else if (yycount == yyargn)
                return 0;
              else
                yyarg[yycount++] = YY_CAST (symbol_kind_type, yyx);
            }
      }

    if (yyarg && yycount == 0 && 0 < yyargn)
      yyarg[0] = symbol_kind::S_YYEMPTY;
    return yycount;
  }






  int
   Parser ::yy_syntax_error_arguments_ (const context& yyctx,
                                                 symbol_kind_type yyarg[], int yyargn) const
  {
    /* There are many possibilities here to consider:
       - If this state is a consistent state with a default action, then
         the only way this function was invoked is if the default action
         is an error action.  In that case, don't check for expected
         tokens because there are none.
       - The only way there can be no lookahead present (in yyla) is
         if this state is a consistent state with a default action.
         Thus, detecting the absence of a lookahead is sufficient to
         determine that there is no unexpected or expected token to
         report.  In that case, just report a simple "syntax error".
       - Don't assume there isn't a lookahead just because this state is
         a consistent state with a default action.  There might have
         been a previous inconsistent state, consistent state with a
         non-default action, or user semantic action that manipulated
         yyla.  (However, yyla is currently not documented for users.)
       - Of course, the expected token list depends on states to have
         correct lookahead information, and it depends on the parser not
         to perform extra reductions after fetching a lookahead from the
         scanner and before detecting a syntax error.  Thus, state merging
         (from LALR or IELR) and default reductions corrupt the expected
         token list.  However, the list is correct for canonical LR with
         one exception: it will still contain any token that will not be
         accepted due to an error action in a later state.
    */

    if (!yyctx.lookahead ().empty ())
      {
        if (yyarg)
          yyarg[0] = yyctx.token ();
        int yyn = yyctx.expected_tokens (yyarg ? yyarg + 1 : yyarg, yyargn - 1);
        return yyn + 1;
      }
    return 0;
  }

  // Generate an error message.
  std::string
   Parser ::yysyntax_error_ (const context& yyctx) const
  {
    // Its maximum.
    enum { YYARGS_MAX = 5 };
    // Arguments of yyformat.
    symbol_kind_type yyarg[YYARGS_MAX];
    int yycount = yy_syntax_error_arguments_ (yyctx, yyarg, YYARGS_MAX);

    char const* yyformat = YY_NULLPTR;
    switch (yycount)
      {
#define YYCASE_(N, S)                         \
        case N:                               \
          yyformat = S;                       \
        break
      default: // Avoid compiler warnings.
        YYCASE_ (0, YY_("syntax error"));
        YYCASE_ (1, YY_("syntax error, unexpected %s"));
        YYCASE_ (2, YY_("syntax error, unexpected %s, expecting %s"));
        YYCASE_ (3, YY_("syntax error, unexpected %s, expecting %s or %s"));
        YYCASE_ (4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
        YYCASE_ (5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
      }

    std::string yyres;
    // Argument number.
    std::ptrdiff_t yyi = 0;
    for (char const* yyp = yyformat; *yyp; ++yyp)
      if (yyp[0] == '%' && yyp[1] == 's' && yyi < yycount)
        {
          yyres += symbol_name (yyarg[yyi++]);
          ++yyp;
        }
      else
        yyres += *yyp;
    return yyres;
  }


  const signed char  Parser ::yypact_ninf_ = -43;

  const signed char  Parser ::yytable_ninf_ = -1;

  const signed char
   Parser ::yypact_[] =
  {
       2,    27,    54,    60,     8,     7,    40,    20,   -43,    17,
      69,     2,   -43,   -43,   -43,   -43,   -43,   -43,   -43,   -43,
     -43,   -43,   -43,   -43,    52,    81,   -43,    82,   -43,    83,
     -43,    94,   -43,    87,    34,    -7,   -43,   -43,   -43,   -43,
     -43,   -43,   -43,    84,    23,    80,   -43,   -43,   -43,    72,
      74,    61,    62,    63,    64,   -43,   -43,   -43,   -43,   -43,
     -43,   -43,    88,    89,   -43,    52,   -43,    49,    77,    85,
     -16,   -43,   -43,   -43,    10,   -43,    -5,    56,   -43,   -43,
     -43,   -43,   -43,   -43,   -43,    76,    52,   -43,   -43,   -13,
      12,   -43,   -43,   -43,   -43,   -43,    78,    90,   -43,    10,
      10,    10,    10,   -43,   -43,    12,    12,   -43,   -43
  };

  const signed char
   Parser ::yydefact_[] =
  {
       3,     0,     0,     0,     0,     0,     0,     0,    22,     0,
       0,     3,     4,     5,    11,    12,    13,    14,     7,     8,
      10,     6,     9,    16,     0,     0,    18,     0,    20,     0,
      21,     0,    39,     0,     0,     0,     1,     2,    29,    30,
      31,    32,    33,     0,    24,     0,    15,    17,    19,     0,
       0,     0,     0,     0,     0,    59,    60,    61,    62,    63,
      41,    40,     0,     0,    23,     0,    26,     0,     0,     0,
       0,    47,    48,    46,     0,    49,     0,     0,    25,    28,
      34,    37,    35,    36,    27,     0,     0,    58,    57,     0,
      53,    56,    44,    45,    43,    42,     0,     0,    50,     0,
       0,     0,     0,    38,    64,    51,    52,    54,    55
  };

  const signed char
   Parser ::yypgoto_[] =
  {
     -43,   107,   -43,   -43,    86,   -43,   -43,   -43,   -43,   -43,
      -2,   -42,   -43,    91,   -32,   -43,   -43,   -43,   -43,   -36,
     -43,   -40,     0,   -43,   -43
  };

  const signed char
   Parser ::yydefgoto_[] =
  {
       0,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      25,    43,    44,    45,    84,    19,    20,    21,    60,    71,
      89,    90,    91,    61,    22
  };

  const signed char
   Parser ::yytable_[] =
  {
      27,    29,    66,    87,    62,    88,     1,     2,     3,     4,
       5,     6,     7,    98,    92,     8,    72,    73,    75,    99,
     100,     9,    24,    78,    38,    39,    40,    41,    42,    87,
      30,    88,    59,    38,    39,    40,    41,    42,    51,    52,
      53,    54,    34,    31,    97,    95,   101,   102,    35,    23,
      27,    29,    65,    96,    24,    55,    33,    56,    57,   105,
     106,    24,    32,    38,    39,    40,    41,    42,    79,    36,
      80,    81,    82,    83,    93,    94,    26,    80,    81,    82,
      83,    24,    28,    23,    26,    28,    30,    24,    70,    70,
      70,    74,    38,    39,    40,    41,    42,    80,    81,    82,
      83,   107,   108,    46,    47,    48,    49,    50,    67,    68,
      64,    69,    86,    76,    77,    85,   104,   103,    37,     0,
      58,     0,     0,     0,     0,     0,    63
  };

  const signed char
   Parser ::yycheck_[] =
  {
       2,     3,    44,    19,    11,    21,     4,     5,     6,     7,
       8,     9,    10,    26,    19,    13,    52,    53,    54,    32,
      33,    19,    27,    65,    40,    41,    42,    43,    44,    19,
      22,    21,    34,    40,    41,    42,    43,    44,     4,     5,
       6,     7,    25,    36,    86,    77,    34,    35,    31,    22,
      52,    53,    29,    85,    27,    21,    36,    23,    24,    99,
     100,    27,    22,    40,    41,    42,    43,    44,    19,     0,
      21,    22,    23,    24,    76,    19,    22,    21,    22,    23,
      24,    27,    22,    22,    22,    22,    22,    27,    27,    27,
      27,    27,    40,    41,    42,    43,    44,    21,    22,    23,
      24,   101,   102,    22,    22,    22,    12,    20,    28,    37,
      26,    37,    27,    25,    25,    38,    26,    39,    11,    -1,
      34,    -1,    -1,    -1,    -1,    -1,    35
  };

  const signed char
   Parser ::yystos_[] =
  {
       0,     4,     5,     6,     7,     8,     9,    10,    13,    19,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    60,
      61,    62,    69,    22,    27,    55,    22,    55,    22,    55,
      22,    36,    22,    36,    25,    31,     0,    46,    40,    41,
      42,    43,    44,    56,    57,    58,    22,    22,    22,    12,
      20,     4,     5,     6,     7,    21,    23,    24,    49,    55,
      63,    68,    11,    58,    26,    29,    56,    28,    37,    37,
      27,    64,    64,    64,    27,    64,    25,    25,    56,    19,
      21,    22,    23,    24,    59,    38,    27,    19,    21,    65,
      66,    67,    19,    55,    19,    59,    59,    56,    26,    32,
      33,    34,    35,    39,    26,    66,    66,    67,    67
  };

  const signed char
   Parser ::yyr1_[] =
  {
       0,    45,    46,    46,    47,    47,    47,    47,    48,    48,
      48,    49,    49,    49,    49,    50,    50,    51,    51,    52,
      52,    53,    54,    55,    56,    56,    56,    57,    57,    58,
      58,    58,    58,    58,    59,    59,    59,    59,    60,    61,
      62,    62,    62,    62,    62,    62,    63,    63,    63,    63,
      64,    65,    65,    65,    66,    66,    66,    67,    67,    68,
      68,    68,    68,    68,    69
  };

  const signed char
   Parser ::yyr2_[] =
  {
       0,     2,     2,     0,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     3,     2,     3,     2,     3,
       2,     2,     1,     3,     1,     3,     2,     3,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     7,     2,
       3,     3,     5,     5,     5,     5,     2,     2,     2,     2,
       3,     3,     3,     1,     3,     3,     1,     1,     1,     1,
       1,     1,     1,     1,     7
  };


#if YYDEBUG || 1
  // YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
  // First, the terminals, then, starting at \a YYNTOKENS, nonterminals.
  const char*
  const  Parser ::yytname_[] =
  {
  "\"end of file\"", "error", "\"invalid token\"", "NEWLINE", "TITRE",
  "SOUS_TITRE", "PARAGRAPHE", "IMAGE", "DEFINE", "TITREPAGE", "STYLE",
  "SELECTSTYLE", "PROPRIETE", "COMMENTAIRE", "SI", "SINON", "FINSI",
  "POUR", "FINI", "IDENTIFIANT", "BLOCS", "ENTIER", "CHAINE",
  "HEX_COULEUR", "RGB_COULEUR", "EGAL", "CROCHET_FERMANT",
  "CROCHET_OUVRANT", "DEUX_POINTS", "VIRGULE", "POINT_VIRGULE", "POINT",
  "PLUS", "MOINS", "MULT", "DIV", "PARENTHESE_OUVRANTE",
  "PARENTHESE_FERMANTE", "ACCOLADE_OUVRANTE", "ACCOLADE_FERMANTE",
  "LARGEUR", "HAUTEUR", "COULEURTEXTE", "COULEURFOND", "OPACITE",
  "$accept", "programme", "programme_element", "declaration",
  "bloc_element", "titre", "sous_titre", "paragraphe", "image",
  "commentaire", "attributs", "liste_attributs", "attribut", "nomattribut",
  "valeur", "define", "titrepage", "variable", "selecteur",
  "index_expression", "expr", "terme", "facteur", "valeurvar", "style", YY_NULLPTR
  };
#endif


#if YYDEBUG
  const short
   Parser ::yyrline_[] =
  {
       0,    72,    72,    73,    77,    78,    79,    80,    84,    85,
      86,    90,    91,    92,    93,    97,   101,   108,   112,   119,
     123,   130,   136,   142,   148,   151,   155,   162,   165,   174,
     175,   176,   177,   178,   182,   183,   184,   185,   189,   196,
     203,   215,   221,   227,   236,   247,   259,   260,   261,   262,
     266,   270,   271,   272,   276,   277,   278,   282,   283,   294,
     295,   296,   297,   298,   302
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
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44
    };
    // Last valid token kind.
    const int code_max = 299;

    if (t <= 0)
      return symbol_kind::S_YYEOF;
    else if (t <= code_max)
      return static_cast <symbol_kind_type> (translate_table[t]);
    else
      return symbol_kind::S_YYUNDEF;
  }

} // yy
#line 2061 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.cpp"

#line 308 "parser/parser.yy"


void yy::Parser::error( const location_type &l, const std::string & err_msg) {
    std::cerr << "Erreur de syntaxe ligne " << l.begin.line 
              << ", colonne " << l.begin.column << ": " << err_msg << std::endl;
}
