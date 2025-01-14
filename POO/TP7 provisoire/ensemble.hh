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
    ensemble(ensemble const& e) : _elements(e._elements) {}
    std::vector<T> get_tab() { return _elements; }
    bool empty() { return _elements.empty(); }
    bool find( T const& e) {
        for (auto i : _elements) {
            if (i == e) {
                return true;
            }
        }
        return false;
    }
    void insert(T const& e) {
        if (!find(e)) {
            _elements.push_back(e);
        }
        else {
            throw std::invalid_argument("Element already in the ensemble");
        }
    }
    void unionwith(ensemble const& e) noexcept {
        for (auto i : e._elements){
            try{
                insert(i);
            }
            catch ( std::invalid_argument const&){
            }
        }
    }
private:
    std::vector<T> _elements;

};
