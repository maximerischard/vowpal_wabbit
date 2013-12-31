#include "sparse_dense.h"
#include "simple_label.h"
#include "gd.h"
#include "rand48.h"

using namespace std;

namespace OT {

  const size_t stride = 5;

#define sign(x)       ( ((x) > 0) ? (1):(-1)) 
  
  const size_t tree = 129;
  const size_t mult_const = 95104348457;

  struct online_tree{
    example synthetic;
    example* original_ec;

    float alpha_mult;
    size_t num_features;
    size_t max_depth;
    size_t current_depth;
    
    vw* all;
    weight* per_feature;

    v_array<feature> out_set;

    feature f;
    float derived_delta;
  };

  const size_t pc = 0;
  const size_t residual_range = 1;
  const size_t residual_variance = 2;
  const size_t residual_regret = 3;
  const size_t delta_loc = 4;

  float* get_entry(online_tree& ot, uint32_t weight_index) {
    return &(ot.per_feature[(weight_index & ot.all->reg.weight_mask) 
	/ ot.all->reg.stride * OT::stride]);
  }

  void clear_cycle(online_tree& ot, uint32_t weight_index) {
    float* entry = get_entry(ot,weight_index);
    uint32_t i = *(uint32_t*)(entry+pc);
    i = i^2;
    entry[pc] = *(float*)&i;
  }
  
  void set_parent(online_tree& ot, uint32_t weight_index) 
  {
    float* entry = get_entry(ot,weight_index);
    uint32_t i = *(uint32_t*)(entry+pc);
    i = i | 1;
    entry[pc] = *(float*)&i;    
  }

  bool parent(online_tree& ot, feature f, float log_delta)
  {
    float* entry = get_entry(ot,f.weight_index);

    uint32_t j = *((uint32_t*)(entry+pc));
    if (j&1) {
#ifdef DEBUG
      cout << "Feature " << f.weight_index << " with value " << f.x 
	   << " is already a parent" << endl;
#endif
      return true;
    }
    else
      {
        entry[delta_loc] = max(entry[delta_loc], log_delta);
        float deviation = 0.;
      
        float delta_alpha_mult = ot.alpha_mult * entry[delta_loc];

        if (entry[residual_variance] * 0.71828 > entry[residual_range]*entry[residual_range]*delta_alpha_mult) 
	  deviation = 2 * sqrt(entry[residual_variance] * 0.71828 * delta_alpha_mult);
        else
	  deviation = 2*entry[residual_range] * max (0.50001, delta_alpha_mult);
      
        if (deviation < 0.0)
	  deviation = 0.0;
      
#ifdef DEBUG2
	cout << "Feature " << f.weight_index << " with residual_variance " << entry[residual_variance] << " residual_range " << entry[residual_range] << " residual_regret " << entry[residual_regret] << " deviation " << deviation << endl;
	cout << "feature " << f.weight_index << " with regret bound " << entry[residual_regret] - deviation << endl;
#endif

        if (entry[residual_regret] - deviation > 0 && ot.all->training)
	  {
#ifdef DEBUG2
	    cout << "setting parent of feature index " << f.weight_index << endl;
#endif
	    set_parent(ot, f.weight_index);
	    return true;
	  }
        else
	  return false;
      }
  }
  
  void update_regret(online_tree &ot, feature f, float base_loss, float feature_loss)
  {
    float* entry = get_entry(ot,f.weight_index);
    float diff = base_loss - feature_loss;

    // cout << "DEBUG: updating regret for " << f.weight_index << endl;
    // cout << "DEBUG: base_loss " << base_loss << endl;
    // cout << "DEBUG: feature_loss " << feature_loss << endl;
    entry[residual_regret] += diff;
    entry[residual_variance] += diff*diff;
    if (entry[residual_range] < fabsf(diff))
      entry[residual_range] = fabsf(diff);

    if (entry[residual_regret] < 0) {
	entry[residual_regret] = 0;
        entry[residual_variance] = 0;
	entry[residual_range] = 0;
    }


    // cout << "DEBUG update_regret: Residual_variance " << entry[residual_variance] << endl;
    // cout << "DEBUG update_regret: Residual_range " << entry[residual_range] << endl;

  }
  
  void set_cycle(online_tree& ot, uint32_t weight_index) 
  {
    float* entry = get_entry(ot,weight_index);
    uint32_t i = *(uint32_t*)(entry+pc);
    i = i | 2;
    entry[pc] = *(float*)&i;    
  }
  
  bool used_feature(online_tree& ot, uint32_t weight_index)
  {
    float* entry = get_entry(ot,weight_index);
    uint32_t i = *(uint32_t*)(entry+pc);
    
    return i & 2;
  }

  void create_new_features(online_tree&, example*, feature);

  inline void create_new_feature(vw& all, void* d, float v, uint32_t u) {
	  online_tree* ot = (online_tree *)d;
    	  feature n;
          n.x = v * ot->f.x;
          if (u == ot->f.weight_index) {
                uint64_t z = ot->f.weight_index;
                n.weight_index =
                        ((size_t)(merand48(z) * all.reg.weight_mask) * all.reg.stride) &
                         all.reg.weight_mask;
          }
          else {
                n.weight_index = ((u ^ ot->f.weight_index ) * mult_const ) & all.reg.weight_mask;
          }

          if (used_feature(*ot, n.weight_index) || n.x == 0.0) return;

          set_cycle(*ot, n.weight_index);

          if (parent(*ot, n, ot->derived_delta)) {
            // cout << "DEBUG recursion, calling create_new_features on feature index " << n.weight_index << " with value " << n.x << endl;
#ifdef DEBUG2
            cout << ec->example_counter << ": New feature " << n.weight_index << " = product of features " << ot->f.weight_index << " and " << u << endl;
#endif
	    feature temp_f = ot->f;
	    float temp_derived_delta = ot->derived_delta;
            create_new_features(*ot, ot->original_ec, n);
	    ot->f = temp_f;
	    ot->derived_delta = temp_derived_delta;
          }
          else {
#ifdef DEBUG
            cout << "Putting a new feature " << n.weight_index << " in out set, product of features " << u << " and " << ot->f.weight_index << endl;
#endif
            ot->out_set.push_back(n);
          }

  }
  
  void create_new_features(online_tree& ot, example* ec, feature f)
  {
  //cout << "DEBUG in create_new_features, feature index = " << f.weight_index << " value = " << f.x << endl;
    float* entry = get_entry(ot,f.weight_index);
    ot.current_depth++;
    if (ot.current_depth > ot.max_depth)
      {
	ot.max_depth = ot.current_depth;
	cout << "new depth: " << ot.max_depth << "\t" << entry[delta_loc] << "\t" << ot.derived_delta << endl;
      }

    ot.synthetic.atomics[tree].push_back(f);
    ot.synthetic.num_features++;
    ot.synthetic.sum_feat_sq[tree] += f.x*f.x;
 
    ot.derived_delta = ot.derived_delta + log(ec->num_features);
    ot.f = f;
    GD::foreach_feature<create_new_feature>(*ot.all, ec, &ot);
    ot.current_depth--;
  }

  void setup_synthetic(online_tree& ot, example* ec)
  {//things to copy
    ot.synthetic.ld = ec->ld;
    ot.synthetic.tag = ec->tag;
    ot.synthetic.example_counter = ec->example_counter;
    ot.synthetic.ft_offset = ec->ft_offset;

    ot.synthetic.test_only = ec->test_only;
    ot.synthetic.end_pass = ec->end_pass;
    ot.synthetic.sorted = ec->sorted;
    ot.synthetic.in_use = ec->in_use;
    ot.synthetic.done = ec->done;
    
    //things to set
    ot.synthetic.atomics[tree].erase();
    ot.synthetic.audit_features[tree].erase();
    ot.synthetic.num_features = 0;
    ot.synthetic.total_sum_feat_sq = 0;
    ot.synthetic.sum_feat_sq[tree] = 0;
    ot.synthetic.example_t = ec->example_t;

    if (ot.synthetic.indices.size()==0)
      ot.synthetic.indices.push_back(tree);
  }
  
  void clear_features(online_tree& ot, v_array<feature>& set)
  {
    feature* begin = set.begin;
    feature* end = set.end;
    for (feature *f = begin; f != end; f++)
      clear_cycle(ot,f->weight_index);
  }
  
  void simple_print_features(vw&all, example *ec)
  {
    // cout << "DEBUG in simple_print_features, size " << ec->atomics[tree].size() << endl;

    for (unsigned char* i = ec->indices.begin; i != ec->indices.end; i++)
      {
        feature* end = ec->atomics[*i].end;
        for (feature* f = ec->atomics[*i].begin; f!= end; f++) {
          cout << "\t" << f->weight_index << ":" << f->x << ":" << all.reg.weight_vector[f->weight_index & all.reg.weight_mask];
        }
      }
    cout << endl;
  }

  
  inline void add_atomic(vw& all, void* d, float v, uint32_t u) {
      online_tree* ot = (online_tree *)d;
      set_cycle(*ot, u);
      feature f = {v,u};
      create_new_features(*ot, ot->original_ec, f);
  }

  void tree_features(online_tree& ot, example* ec)
  {//called with all.reg.stride = 4
    // entry[min_value] is min feature
    // entry[max_value] is max feature
    // entry[pc] is a bit field.  
    //   The first bit is whether the feature can be a parent or not (default not = 0).   
    //   The second bit is used for cycle prevention in composite feature expansion.  1=already used in this example.
    // entry[range_abs] is the absolute value of feature updates
    // entry[update_sum_squared] is the sum of squared update value
    ot.out_set.erase();
    setup_synthetic(ot, ec);

    // simple_print_features(*(ot.all), &ot.synthetic);
#ifdef DEBUG2
    if (ec->example_counter==1) simple_print_features(*(ot.all), ec);
#endif
    
    ot.derived_delta = 0;
    ot.original_ec = ec;
    GD::foreach_feature<add_atomic>(*ot.all, ec, &ot);
    ot.synthetic.total_sum_feat_sq = ot.synthetic.sum_feat_sq[tree];
    
    // cout << "DEBUG: Returning from tree_features, size of synthetic " << ot.synthetic.atomics[tree].size() << endl << endl;

    clear_features(ot, ot.synthetic.atomics[tree]);
    clear_features(ot, ot.out_set);
  }
  
  void learn(void* d, learner& base, example* ec)
  {
    online_tree* ot=(online_tree*)d;
    
    // cout << "---------------- new example, before entering tree features" << endl;
    tree_features(*ot, ec);

    base.learn(&ot->synthetic);
    ot->num_features = ot->synthetic.num_features;

    label_data* ld = (label_data*)(ec->ld);

    ec->final_prediction = ot->synthetic.final_prediction;
    ec->loss = ot->synthetic.loss;

    float base_loss = ot->synthetic.loss;

#ifdef DEBUG2
    cout << "example " << ot->synthetic.example_counter << " count " << ot->synthetic.atomics[tree].size();
    cout << " loss " << ot->synthetic.loss << "\t";
    simple_print_features(*(ot->all), &ot->synthetic);
#endif

    float old_base = ld->initial;
    ld->initial += ec->final_prediction;

    // cout << "final_prediction " << ec->final_prediction << endl;
    // cout << "loss " << ec->loss << endl;
    ot->synthetic.num_features = 1;

#ifdef DEBUG
    cout << "example " << ot->synthetic.example_counter <<
       " outset size "<< ot->out_set.size() << endl;
#endif
    
    for (feature *f = ot->out_set.begin; f != ot->out_set.end; f++)
      {
	ot->synthetic.atomics[tree].erase();
	ot->synthetic.atomics[tree].push_back(*f);

	ot->synthetic.sum_feat_sq[tree] = f->x*f->x;
	ot->synthetic.total_sum_feat_sq = ot->synthetic.sum_feat_sq[tree];

	// cout << "updating regret of feature " << f->weight_index << endl;
	base.learn(&ot->synthetic);

	update_regret(*ot, *f, base_loss, ot->synthetic.loss);
      }

    ld->initial = old_base;
  }

  void finish(void* d)
  {
    online_tree* ot=(online_tree*)d;
    
    ot->synthetic.atomics[tree].delete_v();
    ot->out_set.delete_v();
    
    free(ot->per_feature);
  }

  void finish_online_tree_example(vw& all, void* d, example* ec)
  {
    online_tree *ot = (online_tree *)d;
    size_t temp_num_features = ec->num_features;
    ec->num_features = ot->num_features;
    output_and_account_example(all, ec);
    ec->num_features = temp_num_features;
    VW::finish_example(all,ec);
  }

  
  void save_load(void* data, io_buf& model_file, bool read, bool text)
  {
    online_tree* ot=(online_tree*)data;
    
    if (model_file.files.size() > 0)
      for (size_t i = 0; i < ot->all->length(); i++)
	{ 
	  char buff[512];
	  weight* entry = &(ot->per_feature[stride*i]);
	  uint32_t text_len = sprintf(buff, " parent: %ui residual_range: %f residual_variance: %f residual_regret: %f delta: %f\n", *(uint32_t*)(entry+pc), entry[residual_range], entry[residual_variance], entry[residual_regret], entry[delta_loc]);
	  bin_text_read_write_fixed(model_file, (char*)entry, sizeof(weight)*stride,
				    "", read,
				    buff, text_len, text);
	}
  }

  learner* setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file)
  {
    online_tree* data = (online_tree*)calloc(1,sizeof(online_tree));

    if (vm.count("online_tree"))
      data->alpha_mult = vm["online_tree"].as< float >();
    else
      data->alpha_mult = vm_file["online_tree"].as< float >();

    if (!vm_file.count("online_tree")) 
      {
	std::stringstream ss;
	ss << " --online_tree " << data->alpha_mult;
	all.options_from_file.append(ss.str());
      }

    data->per_feature = (weight*)calloc(all.length()*stride, sizeof(weight));

    for (size_t i = 0; i < all.length(); i++)
      data->per_feature[i*stride + delta_loc] = 1.;

/*
    if (data->alpha_mult <= 0) {
	cerr << "online_tree: parameter should be positive: resetting from " << data->alpha_mult << " to 1" << endl;	
	data->alpha_mult = 1.;
    }
*/

    data->all = &all;
    learner* l = new learner(data, learn, all.l);
    l->set_save_load(save_load);
    l->set_finish(finish);
    l->set_finish_example(finish_online_tree_example);

    return l;

  }
  
}
