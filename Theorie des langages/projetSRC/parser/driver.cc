#include "driver.hh"
#include <iostream>

Driver::Driver(std::shared_ptr<Document> document) : document(document) {}
Driver::~Driver() {}

std::shared_ptr<Document> Driver::getDocument() {
    return document;
}



