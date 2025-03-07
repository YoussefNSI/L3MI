// A Bison parser, made by GNU Bison 3.8.2.

// Skeleton interface for Bison LALR(1) parsers in C++

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


/**
 ** \file /c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.hpp
 ** Define the yy::parser class.
 */

// C++ LALR(1) parser skeleton written by Akim Demaille.

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.

#ifndef YY_YY_C_USERS_RADOU_DOCUMENTS_GITHUB_L3MI_THEORIE_DES_LANGAGES_PROJETSRC_BUILD_PARSER_HPP_INCLUDED
# define YY_YY_C_USERS_RADOU_DOCUMENTS_GITHUB_L3MI_THEORIE_DES_LANGAGES_PROJETSRC_BUILD_PARSER_HPP_INCLUDED
// "%code requires" blocks.
#line 13 "parser/parser.yy"

    #include "Document.h"

    class Scanner;
    class Driver;

    struct TitreInfo{
        int niveau;
    };

    extern std::shared_ptr<Document> doc;


#line 63 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.hpp"

# include <cassert>
# include <cstdlib> // std::abort
# include <iostream>
# include <stdexcept>
# include <string>
# include <vector>

#if defined __cplusplus
# define YY_CPLUSPLUS __cplusplus
#else
# define YY_CPLUSPLUS 199711L
#endif

// Support move semantics when possible.
#if 201103L <= YY_CPLUSPLUS
# define YY_MOVE           std::move
# define YY_MOVE_OR_COPY   move
# define YY_MOVE_REF(Type) Type&&
# define YY_RVREF(Type)    Type&&
# define YY_COPY(Type)     Type
#else
# define YY_MOVE
# define YY_MOVE_OR_COPY   copy
# define YY_MOVE_REF(Type) Type&
# define YY_RVREF(Type)    const Type&
# define YY_COPY(Type)     const Type&
#endif

// Support noexcept when possible.
#if 201103L <= YY_CPLUSPLUS
# define YY_NOEXCEPT noexcept
# define YY_NOTHROW
#else
# define YY_NOEXCEPT
# define YY_NOTHROW throw ()
#endif

// Support constexpr when possible.
#if 201703 <= YY_CPLUSPLUS
# define YY_CONSTEXPR constexpr
#else
# define YY_CONSTEXPR
#endif
# include "location.hh"
#include <typeinfo>
#ifndef YY_ASSERT
# include <cassert>
# define YY_ASSERT assert
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

namespace yy {
#line 203 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.hpp"




  /// A Bison parser.
  class  Parser 
  {
  public:
#ifdef YYSTYPE
# ifdef __GNUC__
#  pragma GCC message "bison: do not #define YYSTYPE in C++, use %define api.value.type"
# endif
    typedef YYSTYPE value_type;
#else
  /// A buffer to store and retrieve objects.
  ///
  /// Sort of a variant, but does not keep track of the nature
  /// of the stored data, since that knowledge is available
  /// via the current parser state.
  class value_type
  {
  public:
    /// Type of *this.
    typedef value_type self_type;

    /// Empty construction.
    value_type () YY_NOEXCEPT
      : yyraw_ ()
      , yytypeid_ (YY_NULLPTR)
    {}

    /// Construct and fill.
    template <typename T>
    value_type (YY_RVREF (T) t)
      : yytypeid_ (&typeid (T))
    {
      YY_ASSERT (sizeof (T) <= size);
      new (yyas_<T> ()) T (YY_MOVE (t));
    }

#if 201103L <= YY_CPLUSPLUS
    /// Non copyable.
    value_type (const self_type&) = delete;
    /// Non copyable.
    self_type& operator= (const self_type&) = delete;
#endif

    /// Destruction, allowed only if empty.
    ~value_type () YY_NOEXCEPT
    {
      YY_ASSERT (!yytypeid_);
    }

# if 201103L <= YY_CPLUSPLUS
    /// Instantiate a \a T in here from \a t.
    template <typename T, typename... U>
    T&
    emplace (U&&... u)
    {
      YY_ASSERT (!yytypeid_);
      YY_ASSERT (sizeof (T) <= size);
      yytypeid_ = & typeid (T);
      return *new (yyas_<T> ()) T (std::forward <U>(u)...);
    }
# else
    /// Instantiate an empty \a T in here.
    template <typename T>
    T&
    emplace ()
    {
      YY_ASSERT (!yytypeid_);
      YY_ASSERT (sizeof (T) <= size);
      yytypeid_ = & typeid (T);
      return *new (yyas_<T> ()) T ();
    }

    /// Instantiate a \a T in here from \a t.
    template <typename T>
    T&
    emplace (const T& t)
    {
      YY_ASSERT (!yytypeid_);
      YY_ASSERT (sizeof (T) <= size);
      yytypeid_ = & typeid (T);
      return *new (yyas_<T> ()) T (t);
    }
# endif

    /// Instantiate an empty \a T in here.
    /// Obsolete, use emplace.
    template <typename T>
    T&
    build ()
    {
      return emplace<T> ();
    }

    /// Instantiate a \a T in here from \a t.
    /// Obsolete, use emplace.
    template <typename T>
    T&
    build (const T& t)
    {
      return emplace<T> (t);
    }

    /// Accessor to a built \a T.
    template <typename T>
    T&
    as () YY_NOEXCEPT
    {
      YY_ASSERT (yytypeid_);
      YY_ASSERT (*yytypeid_ == typeid (T));
      YY_ASSERT (sizeof (T) <= size);
      return *yyas_<T> ();
    }

    /// Const accessor to a built \a T (for %printer).
    template <typename T>
    const T&
    as () const YY_NOEXCEPT
    {
      YY_ASSERT (yytypeid_);
      YY_ASSERT (*yytypeid_ == typeid (T));
      YY_ASSERT (sizeof (T) <= size);
      return *yyas_<T> ();
    }

    /// Swap the content with \a that, of same type.
    ///
    /// Both variants must be built beforehand, because swapping the actual
    /// data requires reading it (with as()), and this is not possible on
    /// unconstructed variants: it would require some dynamic testing, which
    /// should not be the variant's responsibility.
    /// Swapping between built and (possibly) non-built is done with
    /// self_type::move ().
    template <typename T>
    void
    swap (self_type& that) YY_NOEXCEPT
    {
      YY_ASSERT (yytypeid_);
      YY_ASSERT (*yytypeid_ == *that.yytypeid_);
      std::swap (as<T> (), that.as<T> ());
    }

    /// Move the content of \a that to this.
    ///
    /// Destroys \a that.
    template <typename T>
    void
    move (self_type& that)
    {
# if 201103L <= YY_CPLUSPLUS
      emplace<T> (std::move (that.as<T> ()));
# else
      emplace<T> ();
      swap<T> (that);
# endif
      that.destroy<T> ();
    }

# if 201103L <= YY_CPLUSPLUS
    /// Move the content of \a that to this.
    template <typename T>
    void
    move (self_type&& that)
    {
      emplace<T> (std::move (that.as<T> ()));
      that.destroy<T> ();
    }
#endif

    /// Copy the content of \a that to this.
    template <typename T>
    void
    copy (const self_type& that)
    {
      emplace<T> (that.as<T> ());
    }

    /// Destroy the stored \a T.
    template <typename T>
    void
    destroy ()
    {
      as<T> ().~T ();
      yytypeid_ = YY_NULLPTR;
    }

  private:
#if YY_CPLUSPLUS < 201103L
    /// Non copyable.
    value_type (const self_type&);
    /// Non copyable.
    self_type& operator= (const self_type&);
#endif

    /// Accessor to raw memory as \a T.
    template <typename T>
    T*
    yyas_ () YY_NOEXCEPT
    {
      void *yyp = yyraw_;
      return static_cast<T*> (yyp);
     }

    /// Const accessor to raw memory as \a T.
    template <typename T>
    const T*
    yyas_ () const YY_NOEXCEPT
    {
      const void *yyp = yyraw_;
      return static_cast<const T*> (yyp);
     }

    /// An auxiliary type to compute the largest semantic type.
    union union_type
    {
      // TITRE
      // SOUS_TITRE
      char dummy1[sizeof (TitreInfo)];

      // condition
      char dummy2[sizeof (bool)];

      // ENTIER
      // TITRE_INDICE
      // PARAGRAPHE_INDICE
      // IMAGE_INDICE
      // index_expression
      // expr
      // terme
      // facteur
      char dummy3[sizeof (int)];

      // attributs
      // liste_attributs
      // attribut
      char dummy4[sizeof (std::map<std::string, std::string>)];

      // selecteur
      // selecteur_condition
      char dummy5[sizeof (std::pair<std::string, int>)];

      // bloc_element
      // titre
      // sous_titre
      // paragraphe
      // image
      // commentaire
      // titrepage
      // selecteur_variable
      char dummy6[sizeof (std::shared_ptr<Bloc>)];

      // PROPRIETE
      // COMMENTAIRE
      // IDENTIFIANT
      // BLOCS
      // CHAINE
      // HEX_COULEUR
      // RGB_COULEUR
      // nomattribut
      // valeur
      // define
      // style
      char dummy7[sizeof (std::string)];

      // variable
      // valeurvar
      char dummy8[sizeof (std::variant<int, std::string, std::shared_ptr<Bloc>, std::map<std::string, std::string>>)];
    };

    /// The size of the largest semantic type.
    enum { size = sizeof (union_type) };

    /// A buffer to store semantic values.
    union
    {
      /// Strongest alignment constraints.
      long double yyalign_me_;
      /// A buffer large enough to store any of the semantic values.
      char yyraw_[size];
    };

    /// Whether the content is built: if defined, the name of the stored type.
    const std::type_info *yytypeid_;
  };

#endif
    /// Backward compatibility (Bison 3.8).
    typedef value_type semantic_type;

    /// Symbol locations.
    typedef location location_type;

    /// Syntax errors thrown from user actions.
    struct syntax_error : std::runtime_error
    {
      syntax_error (const location_type& l, const std::string& m)
        : std::runtime_error (m)
        , location (l)
      {}

      syntax_error (const syntax_error& s)
        : std::runtime_error (s.what ())
        , location (s.location)
      {}

      ~syntax_error () YY_NOEXCEPT YY_NOTHROW;

      location_type location;
    };

    /// Token kinds.
    struct token
    {
      enum token_kind_type
      {
        YYEMPTY = -2,
    YYEOF = 0,                     // "end of file"
    YYerror = 256,                 // error
    YYUNDEF = 257,                 // "invalid token"
    NEWLINE = 258,                 // NEWLINE
    TITRE = 259,                   // TITRE
    SOUS_TITRE = 260,              // SOUS_TITRE
    PARAGRAPHE = 261,              // PARAGRAPHE
    IMAGE = 262,                   // IMAGE
    DEFINE = 263,                  // DEFINE
    TITREPAGE = 264,               // TITREPAGE
    STYLE = 265,                   // STYLE
    SELECTSTYLE = 266,             // SELECTSTYLE
    PROPRIETE = 267,               // PROPRIETE
    COMMENTAIRE = 268,             // COMMENTAIRE
    SI = 269,                      // SI
    SINON = 270,                   // SINON
    FINSI = 271,                   // FINSI
    POUR = 272,                    // POUR
    FINI = 273,                    // FINI
    IDENTIFIANT = 274,             // IDENTIFIANT
    BLOCS = 275,                   // BLOCS
    ENTIER = 276,                  // ENTIER
    TITRE_INDICE = 277,            // TITRE_INDICE
    PARAGRAPHE_INDICE = 278,       // PARAGRAPHE_INDICE
    IMAGE_INDICE = 279,            // IMAGE_INDICE
    CHAINE = 280,                  // CHAINE
    HEX_COULEUR = 281,             // HEX_COULEUR
    RGB_COULEUR = 282,             // RGB_COULEUR
    EGAL = 283,                    // EGAL
    CROCHET_FERMANT = 284,         // CROCHET_FERMANT
    CROCHET_OUVRANT = 285,         // CROCHET_OUVRANT
    DEUX_POINTS = 286,             // DEUX_POINTS
    VIRGULE = 287,                 // VIRGULE
    POINT_VIRGULE = 288,           // POINT_VIRGULE
    POINT = 289,                   // POINT
    PLUS = 290,                    // PLUS
    MOINS = 291,                   // MOINS
    MULT = 292,                    // MULT
    DIV = 293,                     // DIV
    PARENTHESE_OUVRANTE = 294,     // PARENTHESE_OUVRANTE
    PARENTHESE_FERMANTE = 295,     // PARENTHESE_FERMANTE
    ACCOLADE_OUVRANTE = 296,       // ACCOLADE_OUVRANTE
    ACCOLADE_FERMANTE = 297,       // ACCOLADE_FERMANTE
    LARGEUR = 298,                 // LARGEUR
    HAUTEUR = 299,                 // HAUTEUR
    COULEURTEXTE = 300,            // COULEURTEXTE
    COULEURFOND = 301,             // COULEURFOND
    OPACITE = 302                  // OPACITE
      };
      /// Backward compatibility alias (Bison 3.6).
      typedef token_kind_type yytokentype;
    };

    /// Token kind, as returned by yylex.
    typedef token::token_kind_type token_kind_type;

    /// Backward compatibility alias (Bison 3.6).
    typedef token_kind_type token_type;

    /// Symbol kinds.
    struct symbol_kind
    {
      enum symbol_kind_type
      {
        YYNTOKENS = 48, ///< Number of tokens.
        S_YYEMPTY = -2,
        S_YYEOF = 0,                             // "end of file"
        S_YYerror = 1,                           // error
        S_YYUNDEF = 2,                           // "invalid token"
        S_NEWLINE = 3,                           // NEWLINE
        S_TITRE = 4,                             // TITRE
        S_SOUS_TITRE = 5,                        // SOUS_TITRE
        S_PARAGRAPHE = 6,                        // PARAGRAPHE
        S_IMAGE = 7,                             // IMAGE
        S_DEFINE = 8,                            // DEFINE
        S_TITREPAGE = 9,                         // TITREPAGE
        S_STYLE = 10,                            // STYLE
        S_SELECTSTYLE = 11,                      // SELECTSTYLE
        S_PROPRIETE = 12,                        // PROPRIETE
        S_COMMENTAIRE = 13,                      // COMMENTAIRE
        S_SI = 14,                               // SI
        S_SINON = 15,                            // SINON
        S_FINSI = 16,                            // FINSI
        S_POUR = 17,                             // POUR
        S_FINI = 18,                             // FINI
        S_IDENTIFIANT = 19,                      // IDENTIFIANT
        S_BLOCS = 20,                            // BLOCS
        S_ENTIER = 21,                           // ENTIER
        S_TITRE_INDICE = 22,                     // TITRE_INDICE
        S_PARAGRAPHE_INDICE = 23,                // PARAGRAPHE_INDICE
        S_IMAGE_INDICE = 24,                     // IMAGE_INDICE
        S_CHAINE = 25,                           // CHAINE
        S_HEX_COULEUR = 26,                      // HEX_COULEUR
        S_RGB_COULEUR = 27,                      // RGB_COULEUR
        S_EGAL = 28,                             // EGAL
        S_CROCHET_FERMANT = 29,                  // CROCHET_FERMANT
        S_CROCHET_OUVRANT = 30,                  // CROCHET_OUVRANT
        S_DEUX_POINTS = 31,                      // DEUX_POINTS
        S_VIRGULE = 32,                          // VIRGULE
        S_POINT_VIRGULE = 33,                    // POINT_VIRGULE
        S_POINT = 34,                            // POINT
        S_PLUS = 35,                             // PLUS
        S_MOINS = 36,                            // MOINS
        S_MULT = 37,                             // MULT
        S_DIV = 38,                              // DIV
        S_PARENTHESE_OUVRANTE = 39,              // PARENTHESE_OUVRANTE
        S_PARENTHESE_FERMANTE = 40,              // PARENTHESE_FERMANTE
        S_ACCOLADE_OUVRANTE = 41,                // ACCOLADE_OUVRANTE
        S_ACCOLADE_FERMANTE = 42,                // ACCOLADE_FERMANTE
        S_LARGEUR = 43,                          // LARGEUR
        S_HAUTEUR = 44,                          // HAUTEUR
        S_COULEURTEXTE = 45,                     // COULEURTEXTE
        S_COULEURFOND = 46,                      // COULEURFOND
        S_OPACITE = 47,                          // OPACITE
        S_YYACCEPT = 48,                         // $accept
        S_programme = 49,                        // programme
        S_programme_element = 50,                // programme_element
        S_declaration = 51,                      // declaration
        S_bloc_element = 52,                     // bloc_element
        S_titre = 53,                            // titre
        S_sous_titre = 54,                       // sous_titre
        S_paragraphe = 55,                       // paragraphe
        S_image = 56,                            // image
        S_commentaire = 57,                      // commentaire
        S_attributs = 58,                        // attributs
        S_liste_attributs = 59,                  // liste_attributs
        S_attribut = 60,                         // attribut
        S_nomattribut = 61,                      // nomattribut
        S_valeur = 62,                           // valeur
        S_define = 63,                           // define
        S_titrepage = 64,                        // titrepage
        S_variable = 65,                         // variable
        S_selecteur = 66,                        // selecteur
        S_selecteur2 = 67,                       // selecteur2
        S_selecteur_condition = 68,              // selecteur_condition
        S_selecteur_variable = 69,               // selecteur_variable
        S_index_expression = 70,                 // index_expression
        S_expr = 71,                             // expr
        S_terme = 72,                            // terme
        S_facteur = 73,                          // facteur
        S_valeurvar = 74,                        // valeurvar
        S_style = 75,                            // style
        S_conditionnel = 76,                     // conditionnel
        S_77_1 = 77,                             // $@1
        S_else_clause = 78,                      // else_clause
        S_79_2 = 79,                             // $@2
        S_condition = 80,                        // condition
        S_instructions = 81,                     // instructions
        S_instruction = 82                       // instruction
      };
    };

    /// (Internal) symbol kind.
    typedef symbol_kind::symbol_kind_type symbol_kind_type;

    /// The number of tokens.
    static const symbol_kind_type YYNTOKENS = symbol_kind::YYNTOKENS;

    /// A complete symbol.
    ///
    /// Expects its Base type to provide access to the symbol kind
    /// via kind ().
    ///
    /// Provide access to semantic value and location.
    template <typename Base>
    struct basic_symbol : Base
    {
      /// Alias to Base.
      typedef Base super_type;

      /// Default constructor.
      basic_symbol () YY_NOEXCEPT
        : value ()
        , location ()
      {}

#if 201103L <= YY_CPLUSPLUS
      /// Move constructor.
      basic_symbol (basic_symbol&& that)
        : Base (std::move (that))
        , value ()
        , location (std::move (that.location))
      {
        switch (this->kind ())
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.move< TitreInfo > (std::move (that.value));
        break;

      case symbol_kind::S_condition: // condition
        value.move< bool > (std::move (that.value));
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_TITRE_INDICE: // TITRE_INDICE
      case symbol_kind::S_PARAGRAPHE_INDICE: // PARAGRAPHE_INDICE
      case symbol_kind::S_IMAGE_INDICE: // IMAGE_INDICE
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        value.move< int > (std::move (that.value));
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.move< std::map<std::string, std::string> > (std::move (that.value));
        break;

      case symbol_kind::S_selecteur: // selecteur
      case symbol_kind::S_selecteur_condition: // selecteur_condition
        value.move< std::pair<std::string, int> > (std::move (that.value));
        break;

      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_selecteur_variable: // selecteur_variable
        value.move< std::shared_ptr<Bloc> > (std::move (that.value));
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.move< std::string > (std::move (that.value));
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.move< std::variant<int, std::string, std::shared_ptr<Bloc>, std::map<std::string, std::string>> > (std::move (that.value));
        break;

      default:
        break;
    }

      }
#endif

      /// Copy constructor.
      basic_symbol (const basic_symbol& that);

      /// Constructors for typed symbols.
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, location_type&& l)
        : Base (t)
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const location_type& l)
        : Base (t)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, TitreInfo&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const TitreInfo& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, bool&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const bool& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, int&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const int& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, std::map<std::string, std::string>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const std::map<std::string, std::string>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, std::pair<std::string, int>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const std::pair<std::string, int>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, std::shared_ptr<Bloc>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const std::shared_ptr<Bloc>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, std::string&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const std::string& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, std::variant<int, std::string, std::shared_ptr<Bloc>, std::map<std::string, std::string>>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const std::variant<int, std::string, std::shared_ptr<Bloc>, std::map<std::string, std::string>>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

      /// Destroy the symbol.
      ~basic_symbol ()
      {
        clear ();
      }



      /// Destroy contents, and record that is empty.
      void clear () YY_NOEXCEPT
      {
        // User destructor.
        symbol_kind_type yykind = this->kind ();
        basic_symbol<Base>& yysym = *this;
        (void) yysym;
        switch (yykind)
        {
       default:
          break;
        }

        // Value type destructor.
switch (yykind)
    {
      case symbol_kind::S_TITRE: // TITRE
      case symbol_kind::S_SOUS_TITRE: // SOUS_TITRE
        value.template destroy< TitreInfo > ();
        break;

      case symbol_kind::S_condition: // condition
        value.template destroy< bool > ();
        break;

      case symbol_kind::S_ENTIER: // ENTIER
      case symbol_kind::S_TITRE_INDICE: // TITRE_INDICE
      case symbol_kind::S_PARAGRAPHE_INDICE: // PARAGRAPHE_INDICE
      case symbol_kind::S_IMAGE_INDICE: // IMAGE_INDICE
      case symbol_kind::S_index_expression: // index_expression
      case symbol_kind::S_expr: // expr
      case symbol_kind::S_terme: // terme
      case symbol_kind::S_facteur: // facteur
        value.template destroy< int > ();
        break;

      case symbol_kind::S_attributs: // attributs
      case symbol_kind::S_liste_attributs: // liste_attributs
      case symbol_kind::S_attribut: // attribut
        value.template destroy< std::map<std::string, std::string> > ();
        break;

      case symbol_kind::S_selecteur: // selecteur
      case symbol_kind::S_selecteur_condition: // selecteur_condition
        value.template destroy< std::pair<std::string, int> > ();
        break;

      case symbol_kind::S_bloc_element: // bloc_element
      case symbol_kind::S_titre: // titre
      case symbol_kind::S_sous_titre: // sous_titre
      case symbol_kind::S_paragraphe: // paragraphe
      case symbol_kind::S_image: // image
      case symbol_kind::S_commentaire: // commentaire
      case symbol_kind::S_titrepage: // titrepage
      case symbol_kind::S_selecteur_variable: // selecteur_variable
        value.template destroy< std::shared_ptr<Bloc> > ();
        break;

      case symbol_kind::S_PROPRIETE: // PROPRIETE
      case symbol_kind::S_COMMENTAIRE: // COMMENTAIRE
      case symbol_kind::S_IDENTIFIANT: // IDENTIFIANT
      case symbol_kind::S_BLOCS: // BLOCS
      case symbol_kind::S_CHAINE: // CHAINE
      case symbol_kind::S_HEX_COULEUR: // HEX_COULEUR
      case symbol_kind::S_RGB_COULEUR: // RGB_COULEUR
      case symbol_kind::S_nomattribut: // nomattribut
      case symbol_kind::S_valeur: // valeur
      case symbol_kind::S_define: // define
      case symbol_kind::S_style: // style
        value.template destroy< std::string > ();
        break;

      case symbol_kind::S_variable: // variable
      case symbol_kind::S_valeurvar: // valeurvar
        value.template destroy< std::variant<int, std::string, std::shared_ptr<Bloc>, std::map<std::string, std::string>> > ();
        break;

      default:
        break;
    }

        Base::clear ();
      }

      /// The user-facing name of this symbol.
      std::string name () const YY_NOEXCEPT
      {
        return  Parser ::symbol_name (this->kind ());
      }

      /// Backward compatibility (Bison 3.6).
      symbol_kind_type type_get () const YY_NOEXCEPT;

      /// Whether empty.
      bool empty () const YY_NOEXCEPT;

      /// Destructive move, \a s is emptied into this.
      void move (basic_symbol& s);

      /// The semantic value.
      value_type value;

      /// The location.
      location_type location;

    private:
#if YY_CPLUSPLUS < 201103L
      /// Assignment operator.
      basic_symbol& operator= (const basic_symbol& that);
#endif
    };

    /// Type access provider for token (enum) based symbols.
    struct by_kind
    {
      /// The symbol kind as needed by the constructor.
      typedef token_kind_type kind_type;

      /// Default constructor.
      by_kind () YY_NOEXCEPT;

#if 201103L <= YY_CPLUSPLUS
      /// Move constructor.
      by_kind (by_kind&& that) YY_NOEXCEPT;
#endif

      /// Copy constructor.
      by_kind (const by_kind& that) YY_NOEXCEPT;

      /// Constructor from (external) token numbers.
      by_kind (kind_type t) YY_NOEXCEPT;



      /// Record that this symbol is empty.
      void clear () YY_NOEXCEPT;

      /// Steal the symbol kind from \a that.
      void move (by_kind& that);

      /// The (internal) type number (corresponding to \a type).
      /// \a empty when empty.
      symbol_kind_type kind () const YY_NOEXCEPT;

      /// Backward compatibility (Bison 3.6).
      symbol_kind_type type_get () const YY_NOEXCEPT;

      /// The symbol kind.
      /// \a S_YYEMPTY when empty.
      symbol_kind_type kind_;
    };

    /// Backward compatibility for a private implementation detail (Bison 3.6).
    typedef by_kind by_type;

    /// "External" symbols: returned by the scanner.
    struct symbol_type : basic_symbol<by_kind>
    {
      /// Superclass.
      typedef basic_symbol<by_kind> super_type;

      /// Empty symbol.
      symbol_type () YY_NOEXCEPT {}

      /// Constructor for valueless symbols, and symbols from each type.
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, location_type l)
        : super_type (token_kind_type (tok), std::move (l))
#else
      symbol_type (int tok, const location_type& l)
        : super_type (token_kind_type (tok), l)
#endif
      {
#if !defined _MSC_VER || defined __clang__
        YY_ASSERT (tok == token::YYEOF
                   || (token::YYerror <= tok && tok <= token::NEWLINE)
                   || (token::PARAGRAPHE <= tok && tok <= token::SELECTSTYLE)
                   || (token::SI <= tok && tok <= token::FINI)
                   || (token::EGAL <= tok && tok <= token::OPACITE));
#endif
      }
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, TitreInfo v, location_type l)
        : super_type (token_kind_type (tok), std::move (v), std::move (l))
#else
      symbol_type (int tok, const TitreInfo& v, const location_type& l)
        : super_type (token_kind_type (tok), v, l)
#endif
      {
#if !defined _MSC_VER || defined __clang__
        YY_ASSERT ((token::TITRE <= tok && tok <= token::SOUS_TITRE));
#endif
      }
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, int v, location_type l)
        : super_type (token_kind_type (tok), std::move (v), std::move (l))
#else
      symbol_type (int tok, const int& v, const location_type& l)
        : super_type (token_kind_type (tok), v, l)
#endif
      {
#if !defined _MSC_VER || defined __clang__
        YY_ASSERT ((token::ENTIER <= tok && tok <= token::IMAGE_INDICE));
#endif
      }
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, std::string v, location_type l)
        : super_type (token_kind_type (tok), std::move (v), std::move (l))
#else
      symbol_type (int tok, const std::string& v, const location_type& l)
        : super_type (token_kind_type (tok), v, l)
#endif
      {
#if !defined _MSC_VER || defined __clang__
        YY_ASSERT ((token::PROPRIETE <= tok && tok <= token::COMMENTAIRE)
                   || (token::IDENTIFIANT <= tok && tok <= token::BLOCS)
                   || (token::CHAINE <= tok && tok <= token::RGB_COULEUR));
#endif
      }
    };

    /// Build a parser object.
     Parser  (Scanner &scanner_yyarg, Driver &driver_yyarg);
    virtual ~ Parser  ();

#if 201103L <= YY_CPLUSPLUS
    /// Non copyable.
     Parser  (const  Parser &) = delete;
    /// Non copyable.
     Parser & operator= (const  Parser &) = delete;
#endif

    /// Parse.  An alias for parse ().
    /// \returns  0 iff parsing succeeded.
    int operator() ();

    /// Parse.
    /// \returns  0 iff parsing succeeded.
    virtual int parse ();

#if YYDEBUG
    /// The current debugging stream.
    std::ostream& debug_stream () const YY_ATTRIBUTE_PURE;
    /// Set the current debugging stream.
    void set_debug_stream (std::ostream &);

    /// Type for debugging levels.
    typedef int debug_level_type;
    /// The current debugging level.
    debug_level_type debug_level () const YY_ATTRIBUTE_PURE;
    /// Set the current debugging level.
    void set_debug_level (debug_level_type l);
#endif

    /// Report a syntax error.
    /// \param loc    where the syntax error is found.
    /// \param msg    a description of the syntax error.
    virtual void error (const location_type& loc, const std::string& msg);

    /// Report a syntax error.
    void error (const syntax_error& err);

    /// The user-facing name of the symbol whose (internal) number is
    /// YYSYMBOL.  No bounds checking.
    static std::string symbol_name (symbol_kind_type yysymbol);

    // Implementation of make_symbol for each token kind.
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_YYEOF (location_type l)
      {
        return symbol_type (token::YYEOF, std::move (l));
      }
#else
      static
      symbol_type
      make_YYEOF (const location_type& l)
      {
        return symbol_type (token::YYEOF, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_YYerror (location_type l)
      {
        return symbol_type (token::YYerror, std::move (l));
      }
#else
      static
      symbol_type
      make_YYerror (const location_type& l)
      {
        return symbol_type (token::YYerror, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_YYUNDEF (location_type l)
      {
        return symbol_type (token::YYUNDEF, std::move (l));
      }
#else
      static
      symbol_type
      make_YYUNDEF (const location_type& l)
      {
        return symbol_type (token::YYUNDEF, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NEWLINE (location_type l)
      {
        return symbol_type (token::NEWLINE, std::move (l));
      }
#else
      static
      symbol_type
      make_NEWLINE (const location_type& l)
      {
        return symbol_type (token::NEWLINE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TITRE (TitreInfo v, location_type l)
      {
        return symbol_type (token::TITRE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TITRE (const TitreInfo& v, const location_type& l)
      {
        return symbol_type (token::TITRE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SOUS_TITRE (TitreInfo v, location_type l)
      {
        return symbol_type (token::SOUS_TITRE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SOUS_TITRE (const TitreInfo& v, const location_type& l)
      {
        return symbol_type (token::SOUS_TITRE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARAGRAPHE (location_type l)
      {
        return symbol_type (token::PARAGRAPHE, std::move (l));
      }
#else
      static
      symbol_type
      make_PARAGRAPHE (const location_type& l)
      {
        return symbol_type (token::PARAGRAPHE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IMAGE (location_type l)
      {
        return symbol_type (token::IMAGE, std::move (l));
      }
#else
      static
      symbol_type
      make_IMAGE (const location_type& l)
      {
        return symbol_type (token::IMAGE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DEFINE (location_type l)
      {
        return symbol_type (token::DEFINE, std::move (l));
      }
#else
      static
      symbol_type
      make_DEFINE (const location_type& l)
      {
        return symbol_type (token::DEFINE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TITREPAGE (location_type l)
      {
        return symbol_type (token::TITREPAGE, std::move (l));
      }
#else
      static
      symbol_type
      make_TITREPAGE (const location_type& l)
      {
        return symbol_type (token::TITREPAGE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STYLE (location_type l)
      {
        return symbol_type (token::STYLE, std::move (l));
      }
#else
      static
      symbol_type
      make_STYLE (const location_type& l)
      {
        return symbol_type (token::STYLE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SELECTSTYLE (location_type l)
      {
        return symbol_type (token::SELECTSTYLE, std::move (l));
      }
#else
      static
      symbol_type
      make_SELECTSTYLE (const location_type& l)
      {
        return symbol_type (token::SELECTSTYLE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PROPRIETE (std::string v, location_type l)
      {
        return symbol_type (token::PROPRIETE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PROPRIETE (const std::string& v, const location_type& l)
      {
        return symbol_type (token::PROPRIETE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMMENTAIRE (std::string v, location_type l)
      {
        return symbol_type (token::COMMENTAIRE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMMENTAIRE (const std::string& v, const location_type& l)
      {
        return symbol_type (token::COMMENTAIRE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SI (location_type l)
      {
        return symbol_type (token::SI, std::move (l));
      }
#else
      static
      symbol_type
      make_SI (const location_type& l)
      {
        return symbol_type (token::SI, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SINON (location_type l)
      {
        return symbol_type (token::SINON, std::move (l));
      }
#else
      static
      symbol_type
      make_SINON (const location_type& l)
      {
        return symbol_type (token::SINON, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FINSI (location_type l)
      {
        return symbol_type (token::FINSI, std::move (l));
      }
#else
      static
      symbol_type
      make_FINSI (const location_type& l)
      {
        return symbol_type (token::FINSI, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_POUR (location_type l)
      {
        return symbol_type (token::POUR, std::move (l));
      }
#else
      static
      symbol_type
      make_POUR (const location_type& l)
      {
        return symbol_type (token::POUR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FINI (location_type l)
      {
        return symbol_type (token::FINI, std::move (l));
      }
#else
      static
      symbol_type
      make_FINI (const location_type& l)
      {
        return symbol_type (token::FINI, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IDENTIFIANT (std::string v, location_type l)
      {
        return symbol_type (token::IDENTIFIANT, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IDENTIFIANT (const std::string& v, const location_type& l)
      {
        return symbol_type (token::IDENTIFIANT, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BLOCS (std::string v, location_type l)
      {
        return symbol_type (token::BLOCS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BLOCS (const std::string& v, const location_type& l)
      {
        return symbol_type (token::BLOCS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENTIER (int v, location_type l)
      {
        return symbol_type (token::ENTIER, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ENTIER (const int& v, const location_type& l)
      {
        return symbol_type (token::ENTIER, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TITRE_INDICE (int v, location_type l)
      {
        return symbol_type (token::TITRE_INDICE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TITRE_INDICE (const int& v, const location_type& l)
      {
        return symbol_type (token::TITRE_INDICE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARAGRAPHE_INDICE (int v, location_type l)
      {
        return symbol_type (token::PARAGRAPHE_INDICE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PARAGRAPHE_INDICE (const int& v, const location_type& l)
      {
        return symbol_type (token::PARAGRAPHE_INDICE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IMAGE_INDICE (int v, location_type l)
      {
        return symbol_type (token::IMAGE_INDICE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IMAGE_INDICE (const int& v, const location_type& l)
      {
        return symbol_type (token::IMAGE_INDICE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHAINE (std::string v, location_type l)
      {
        return symbol_type (token::CHAINE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CHAINE (const std::string& v, const location_type& l)
      {
        return symbol_type (token::CHAINE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HEX_COULEUR (std::string v, location_type l)
      {
        return symbol_type (token::HEX_COULEUR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HEX_COULEUR (const std::string& v, const location_type& l)
      {
        return symbol_type (token::HEX_COULEUR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RGB_COULEUR (std::string v, location_type l)
      {
        return symbol_type (token::RGB_COULEUR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RGB_COULEUR (const std::string& v, const location_type& l)
      {
        return symbol_type (token::RGB_COULEUR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EGAL (location_type l)
      {
        return symbol_type (token::EGAL, std::move (l));
      }
#else
      static
      symbol_type
      make_EGAL (const location_type& l)
      {
        return symbol_type (token::EGAL, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CROCHET_FERMANT (location_type l)
      {
        return symbol_type (token::CROCHET_FERMANT, std::move (l));
      }
#else
      static
      symbol_type
      make_CROCHET_FERMANT (const location_type& l)
      {
        return symbol_type (token::CROCHET_FERMANT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CROCHET_OUVRANT (location_type l)
      {
        return symbol_type (token::CROCHET_OUVRANT, std::move (l));
      }
#else
      static
      symbol_type
      make_CROCHET_OUVRANT (const location_type& l)
      {
        return symbol_type (token::CROCHET_OUVRANT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DEUX_POINTS (location_type l)
      {
        return symbol_type (token::DEUX_POINTS, std::move (l));
      }
#else
      static
      symbol_type
      make_DEUX_POINTS (const location_type& l)
      {
        return symbol_type (token::DEUX_POINTS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VIRGULE (location_type l)
      {
        return symbol_type (token::VIRGULE, std::move (l));
      }
#else
      static
      symbol_type
      make_VIRGULE (const location_type& l)
      {
        return symbol_type (token::VIRGULE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_POINT_VIRGULE (location_type l)
      {
        return symbol_type (token::POINT_VIRGULE, std::move (l));
      }
#else
      static
      symbol_type
      make_POINT_VIRGULE (const location_type& l)
      {
        return symbol_type (token::POINT_VIRGULE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_POINT (location_type l)
      {
        return symbol_type (token::POINT, std::move (l));
      }
#else
      static
      symbol_type
      make_POINT (const location_type& l)
      {
        return symbol_type (token::POINT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PLUS (location_type l)
      {
        return symbol_type (token::PLUS, std::move (l));
      }
#else
      static
      symbol_type
      make_PLUS (const location_type& l)
      {
        return symbol_type (token::PLUS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MOINS (location_type l)
      {
        return symbol_type (token::MOINS, std::move (l));
      }
#else
      static
      symbol_type
      make_MOINS (const location_type& l)
      {
        return symbol_type (token::MOINS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MULT (location_type l)
      {
        return symbol_type (token::MULT, std::move (l));
      }
#else
      static
      symbol_type
      make_MULT (const location_type& l)
      {
        return symbol_type (token::MULT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DIV (location_type l)
      {
        return symbol_type (token::DIV, std::move (l));
      }
#else
      static
      symbol_type
      make_DIV (const location_type& l)
      {
        return symbol_type (token::DIV, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARENTHESE_OUVRANTE (location_type l)
      {
        return symbol_type (token::PARENTHESE_OUVRANTE, std::move (l));
      }
#else
      static
      symbol_type
      make_PARENTHESE_OUVRANTE (const location_type& l)
      {
        return symbol_type (token::PARENTHESE_OUVRANTE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARENTHESE_FERMANTE (location_type l)
      {
        return symbol_type (token::PARENTHESE_FERMANTE, std::move (l));
      }
#else
      static
      symbol_type
      make_PARENTHESE_FERMANTE (const location_type& l)
      {
        return symbol_type (token::PARENTHESE_FERMANTE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ACCOLADE_OUVRANTE (location_type l)
      {
        return symbol_type (token::ACCOLADE_OUVRANTE, std::move (l));
      }
#else
      static
      symbol_type
      make_ACCOLADE_OUVRANTE (const location_type& l)
      {
        return symbol_type (token::ACCOLADE_OUVRANTE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ACCOLADE_FERMANTE (location_type l)
      {
        return symbol_type (token::ACCOLADE_FERMANTE, std::move (l));
      }
#else
      static
      symbol_type
      make_ACCOLADE_FERMANTE (const location_type& l)
      {
        return symbol_type (token::ACCOLADE_FERMANTE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LARGEUR (location_type l)
      {
        return symbol_type (token::LARGEUR, std::move (l));
      }
#else
      static
      symbol_type
      make_LARGEUR (const location_type& l)
      {
        return symbol_type (token::LARGEUR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HAUTEUR (location_type l)
      {
        return symbol_type (token::HAUTEUR, std::move (l));
      }
#else
      static
      symbol_type
      make_HAUTEUR (const location_type& l)
      {
        return symbol_type (token::HAUTEUR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COULEURTEXTE (location_type l)
      {
        return symbol_type (token::COULEURTEXTE, std::move (l));
      }
#else
      static
      symbol_type
      make_COULEURTEXTE (const location_type& l)
      {
        return symbol_type (token::COULEURTEXTE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COULEURFOND (location_type l)
      {
        return symbol_type (token::COULEURFOND, std::move (l));
      }
#else
      static
      symbol_type
      make_COULEURFOND (const location_type& l)
      {
        return symbol_type (token::COULEURFOND, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OPACITE (location_type l)
      {
        return symbol_type (token::OPACITE, std::move (l));
      }
#else
      static
      symbol_type
      make_OPACITE (const location_type& l)
      {
        return symbol_type (token::OPACITE, l);
      }
#endif


    class context
    {
    public:
      context (const  Parser & yyparser, const symbol_type& yyla);
      const symbol_type& lookahead () const YY_NOEXCEPT { return yyla_; }
      symbol_kind_type token () const YY_NOEXCEPT { return yyla_.kind (); }
      const location_type& location () const YY_NOEXCEPT { return yyla_.location; }

      /// Put in YYARG at most YYARGN of the expected tokens, and return the
      /// number of tokens stored in YYARG.  If YYARG is null, return the
      /// number of expected tokens (guaranteed to be less than YYNTOKENS).
      int expected_tokens (symbol_kind_type yyarg[], int yyargn) const;

    private:
      const  Parser & yyparser_;
      const symbol_type& yyla_;
    };

  private:
#if YY_CPLUSPLUS < 201103L
    /// Non copyable.
     Parser  (const  Parser &);
    /// Non copyable.
     Parser & operator= (const  Parser &);
#endif


    /// Stored state numbers (used for stacks).
    typedef unsigned char state_type;

    /// The arguments of the error message.
    int yy_syntax_error_arguments_ (const context& yyctx,
                                    symbol_kind_type yyarg[], int yyargn) const;

    /// Generate an error message.
    /// \param yyctx     the context in which the error occurred.
    virtual std::string yysyntax_error_ (const context& yyctx) const;
    /// Compute post-reduction state.
    /// \param yystate   the current state
    /// \param yysym     the nonterminal to push on the stack
    static state_type yy_lr_goto_state_ (state_type yystate, int yysym);

    /// Whether the given \c yypact_ value indicates a defaulted state.
    /// \param yyvalue   the value to check
    static bool yy_pact_value_is_default_ (int yyvalue) YY_NOEXCEPT;

    /// Whether the given \c yytable_ value indicates a syntax error.
    /// \param yyvalue   the value to check
    static bool yy_table_value_is_error_ (int yyvalue) YY_NOEXCEPT;

    static const short yypact_ninf_;
    static const signed char yytable_ninf_;

    /// Convert a scanner token kind \a t to a symbol kind.
    /// In theory \a t should be a token_kind_type, but character literals
    /// are valid, yet not members of the token_kind_type enum.
    static symbol_kind_type yytranslate_ (int t) YY_NOEXCEPT;

    /// Convert the symbol name \a n to a form suitable for a diagnostic.
    static std::string yytnamerr_ (const char *yystr);

    /// For a symbol, its name in clear.
    static const char* const yytname_[];


    // Tables.
    // YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
    // STATE-NUM.
    static const short yypact_[];

    // YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
    // Performed when YYTABLE does not specify something else to do.  Zero
    // means the default is an error.
    static const signed char yydefact_[];

    // YYPGOTO[NTERM-NUM].
    static const short yypgoto_[];

    // YYDEFGOTO[NTERM-NUM].
    static const unsigned char yydefgoto_[];

    // YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
    // positive, shift that token.  If negative, reduce the rule whose
    // number is the opposite.  If YYTABLE_NINF, syntax error.
    static const unsigned char yytable_[];

    static const short yycheck_[];

    // YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
    // state STATE-NUM.
    static const signed char yystos_[];

    // YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.
    static const signed char yyr1_[];

    // YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.
    static const signed char yyr2_[];


#if YYDEBUG
    // YYRLINE[YYN] -- Source line where rule number YYN was defined.
    static const short yyrline_[];
    /// Report on the debug stream that the rule \a r is going to be reduced.
    virtual void yy_reduce_print_ (int r) const;
    /// Print the state stack on the debug stream.
    virtual void yy_stack_print_ () const;

    /// Debugging level.
    int yydebug_;
    /// Debug stream.
    std::ostream* yycdebug_;

    /// \brief Display a symbol kind, value and location.
    /// \param yyo    The output stream.
    /// \param yysym  The symbol.
    template <typename Base>
    void yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const;
#endif

    /// \brief Reclaim the memory associated to a symbol.
    /// \param yymsg     Why this token is reclaimed.
    ///                  If null, print nothing.
    /// \param yysym     The symbol.
    template <typename Base>
    void yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const;

  private:
    /// Type access provider for state based symbols.
    struct by_state
    {
      /// Default constructor.
      by_state () YY_NOEXCEPT;

      /// The symbol kind as needed by the constructor.
      typedef state_type kind_type;

      /// Constructor.
      by_state (kind_type s) YY_NOEXCEPT;

      /// Copy constructor.
      by_state (const by_state& that) YY_NOEXCEPT;

      /// Record that this symbol is empty.
      void clear () YY_NOEXCEPT;

      /// Steal the symbol kind from \a that.
      void move (by_state& that);

      /// The symbol kind (corresponding to \a state).
      /// \a symbol_kind::S_YYEMPTY when empty.
      symbol_kind_type kind () const YY_NOEXCEPT;

      /// The state number used to denote an empty symbol.
      /// We use the initial state, as it does not have a value.
      enum { empty_state = 0 };

      /// The state.
      /// \a empty when empty.
      state_type state;
    };

    /// "Internal" symbol: element of the stack.
    struct stack_symbol_type : basic_symbol<by_state>
    {
      /// Superclass.
      typedef basic_symbol<by_state> super_type;
      /// Construct an empty symbol.
      stack_symbol_type ();
      /// Move or copy construction.
      stack_symbol_type (YY_RVREF (stack_symbol_type) that);
      /// Steal the contents from \a sym to build this.
      stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) sym);
#if YY_CPLUSPLUS < 201103L
      /// Assignment, needed by push_back by some old implementations.
      /// Moves the contents of that.
      stack_symbol_type& operator= (stack_symbol_type& that);

      /// Assignment, needed by push_back by other implementations.
      /// Needed by some other old implementations.
      stack_symbol_type& operator= (const stack_symbol_type& that);
#endif
    };

    /// A stack with random access from its top.
    template <typename T, typename S = std::vector<T> >
    class stack
    {
    public:
      // Hide our reversed order.
      typedef typename S::iterator iterator;
      typedef typename S::const_iterator const_iterator;
      typedef typename S::size_type size_type;
      typedef typename std::ptrdiff_t index_type;

      stack (size_type n = 200) YY_NOEXCEPT
        : seq_ (n)
      {}

#if 201103L <= YY_CPLUSPLUS
      /// Non copyable.
      stack (const stack&) = delete;
      /// Non copyable.
      stack& operator= (const stack&) = delete;
#endif

      /// Random access.
      ///
      /// Index 0 returns the topmost element.
      const T&
      operator[] (index_type i) const
      {
        return seq_[size_type (size () - 1 - i)];
      }

      /// Random access.
      ///
      /// Index 0 returns the topmost element.
      T&
      operator[] (index_type i)
      {
        return seq_[size_type (size () - 1 - i)];
      }

      /// Steal the contents of \a t.
      ///
      /// Close to move-semantics.
      void
      push (YY_MOVE_REF (T) t)
      {
        seq_.push_back (T ());
        operator[] (0).move (t);
      }

      /// Pop elements from the stack.
      void
      pop (std::ptrdiff_t n = 1) YY_NOEXCEPT
      {
        for (; 0 < n; --n)
          seq_.pop_back ();
      }

      /// Pop all elements from the stack.
      void
      clear () YY_NOEXCEPT
      {
        seq_.clear ();
      }

      /// Number of elements on the stack.
      index_type
      size () const YY_NOEXCEPT
      {
        return index_type (seq_.size ());
      }

      /// Iterator on top of the stack (going downwards).
      const_iterator
      begin () const YY_NOEXCEPT
      {
        return seq_.begin ();
      }

      /// Bottom of the stack.
      const_iterator
      end () const YY_NOEXCEPT
      {
        return seq_.end ();
      }

      /// Present a slice of the top of a stack.
      class slice
      {
      public:
        slice (const stack& stack, index_type range) YY_NOEXCEPT
          : stack_ (stack)
          , range_ (range)
        {}

        const T&
        operator[] (index_type i) const
        {
          return stack_[range_ - i];
        }

      private:
        const stack& stack_;
        index_type range_;
      };

    private:
#if YY_CPLUSPLUS < 201103L
      /// Non copyable.
      stack (const stack&);
      /// Non copyable.
      stack& operator= (const stack&);
#endif
      /// The wrapped container.
      S seq_;
    };


    /// Stack type.
    typedef stack<stack_symbol_type> stack_type;

    /// The stack.
    stack_type yystack_;

    /// Push a new state on the stack.
    /// \param m    a debug message to display
    ///             if null, no trace is output.
    /// \param sym  the symbol
    /// \warning the contents of \a s.value is stolen.
    void yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym);

    /// Push a new look ahead token on the state on the stack.
    /// \param m    a debug message to display
    ///             if null, no trace is output.
    /// \param s    the state
    /// \param sym  the symbol (for its value and location).
    /// \warning the contents of \a sym.value is stolen.
    void yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym);

    /// Pop \a n symbols from the stack.
    void yypop_ (int n = 1) YY_NOEXCEPT;

    /// Constants.
    enum
    {
      yylast_ = 170,     ///< Last index in yytable_.
      yynnts_ = 35,  ///< Number of nonterminal symbols.
      yyfinal_ = 46 ///< Termination state number.
    };


    // User arguments.
    Scanner &scanner;
    Driver &driver;

  };


} // yy
#line 2243 "/c/Users/radou/Documents/GitHub/L3MI/Theorie des langages/projetSRC/build/parser.hpp"




#endif // !YY_YY_C_USERS_RADOU_DOCUMENTS_GITHUB_L3MI_THEORIE_DES_LANGAGES_PROJETSRC_BUILD_PARSER_HPP_INCLUDED
