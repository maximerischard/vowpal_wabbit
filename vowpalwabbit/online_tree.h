#ifndef OT_H
#define OT_H

namespace OT {
  learner* setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file);
}
#endif
