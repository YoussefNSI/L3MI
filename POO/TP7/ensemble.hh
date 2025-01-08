#ifndef ENSEMBLE_HH
#define ENSEMBLE_HH

#include <stdexcept>
#include <vector>

#endif // ENSEMBLE_HH


template<typename T>

class ensemble {
public:
    ensemble() {}
    ~ensemble() {};
    ensemble(const ensemble& e) : _elements(e._elements) {}
    bool empty() { return _elements.empty(); }
    bool find(const T e) {
        for (auto i : _elements) {
            if (i == e) {
                return true;
            }
        }
    }
    void insert(const T e) {
        if (!find(e)) {
            _elements.push_back(e);
        }
        else {
            throw std::invalid_argument("Element already in the ensemble");
        }
    }
    void unionwith(const ensemble& e) noexcept {
        for (auto i : e._elements){
            insert(i);
        }
    }
private:
    std::vector<T> _elements;

};
