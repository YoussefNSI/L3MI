#include <iostream>
#include <vector>
#include <memory>


struct Node {
    int gain;
    int visits;
    Node* parent;
    std::vector<std::unique_ptr<Node>> children;
    
}