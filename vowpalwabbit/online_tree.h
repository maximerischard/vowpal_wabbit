#ifndef ONLINE_TREE_H
#define ONLINE_TREE_H

namespace ONLINE_TREE {
  LEARNER::learner* setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file);
}
#endif
