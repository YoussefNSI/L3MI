#ifndef IMAGE_H
#define IMAGE_H

#include "bloc.h"

class Image : public Bloc {
    std::string src;
public:
    Image(const std::string& src) : src(src) {}
    std::string genererHTML() const override {
        return "<img src=\"" + src + "\" alt=\"Image\">";
    }
};

#endif // IMAGE_H