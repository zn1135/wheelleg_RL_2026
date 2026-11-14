/* 仅PTY集成检查替换GPU计算；实际main/串口/日志/观测不替换。 */
#include "inference.hpp"
#include <fstream>
#include <sstream>
namespace jrl {
std::string sha256(const std::string&) { return "test-double-not-a-model-hash"; }
struct Inference::Impl {};
Inference::Inference(const std::string&):impl_(new Impl) {}
Inference::~Inference()=default;
Action Inference::run(const Input& x) { Action a;for(unsigned i=0;i<6;i++) a[i]=x[i]*.01f;return a; }
void Inference::convert(const std::string&,const std::string&) { throw std::runtime_error("test executable cannot convert models"); }
}
