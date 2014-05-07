#include <float.h>

#include "reductions.h"
#include "simple_label.h"
#include "gd.h"
#include "rand48.h"
#include "accumulate.h"

using namespace std;
using namespace LEARNER;

namespace ONLINE_TREE {

#define sign(x)       ( ((x) > 0) ? (1):(-1)) 
  
  const size_t tree = 129;
  const size_t mult_const = 95104348457;

  struct feature_stats{
    uint32_t parent;
    float range;
    float variance;
    float value;
    float delta;
  };

  struct online_tree{
    example synthetic;
    example* original_ec;

    float alpha_mult;
    size_t num_features;
    size_t max_depth;
    size_t current_depth;

    size_t base_period;
    float period_power;

    size_t next;

    float best_score_in_period;
    float best_weight_in_period;

    float best_previous_score;
    float best_previous_weight;
    
    vw* all;
    feature_stats* per_feature;
    float* prev_variance;
    float* prev_value;

    v_array<feature> out_set;

    feature f;
    float derived_delta;
  };


  void reduce_fstats(feature_stats& s1, const feature_stats& s2) {
    s1.parent |= s2.parent;
    s1.range = max(s1.range, s2.range);
    s1.variance += s2.variance;
    s1.value += s2.value;
    s1.delta = max(s1.delta, s2.delta);
  }

  void allreduce_fstats(online_tree& ot) {
    vw& all = *ot.all;

    if(all.span_server == "") 
      return;

    //cerr<<"In allreduce_fstats\n";
    for(uint32_t i = 0;i < all.length(); i++) {
	ot.per_feature[i].variance -= ot.prev_variance[i];
	ot.per_feature[i].value -= ot.prev_value[i];
    }

    all_reduce<feature_stats, reduce_fstats>(ot.per_feature, all.total, all.span_server, all.unique_id, all.total, all.node, all.socks);

    for(uint32_t i = 0;i < all.length(); i++) {
      ot.per_feature[i].variance += ot.prev_variance[i];
      ot.prev_variance[i] = ot.per_feature[i].variance;
      ot.per_feature[i].value += ot.prev_value[i];
      ot.prev_value[i] = ot.per_feature[i].value;
    }
  }



  feature_stats* get_entry(online_tree& ot, uint32_t weight_index) {
    return &(ot.per_feature[((weight_index & ot.all->reg.weight_mask) 
			     >> ot.all->reg.stride_shift)]);
  }

  void clear_cycle(online_tree& ot, uint32_t weight_index) {
    feature_stats* entry = get_entry(ot,weight_index);
    entry->parent ^= 2;
  }

  float get_deviation(float range, float variance, float alpha_mult, float log_delta)
  {
        float deviation = 0.;
      
        float delta_alpha_mult = alpha_mult * log_delta;
	
        if (variance * 0.71828 > range*range*delta_alpha_mult) 
	  deviation = 2 * sqrt(variance * 0.71828 * delta_alpha_mult);
        else
	  deviation = 2 * range * max((float)0.50001, delta_alpha_mult);
      
        if (deviation < 0.0)
	  deviation = 0.0;    
	return deviation;
  }
  
  bool inset(online_tree& ot, feature f, float log_delta)
  {    
    feature_stats* entry = get_entry(ot,f.weight_index);

    if (log_delta <= entry->delta)
      {
	//	cout << "parent by entry " << log_delta << endl;	
	return true;
      }
    else
      {
	if(ot.best_previous_score == FLT_MAX) return false;

	float deviation = get_deviation(entry->range, entry->variance, ot.alpha_mult, log_delta);
      
#ifdef DEBUG2
	cout << "Feature " << f.weight_index << " with residual_variance " << entry->variance << " residual_range " << entry->range << " residual_regret " << entry->value << " deviation " << deviation << endl;
	cout << "feature " << f.weight_index << " with regret bound " << entry->value - deviation << endl;
#endif
	float diff = entry->value - deviation;
	if (diff > ot.best_score_in_period)
	  ot.best_score_in_period = diff;	

        if (diff > ot.best_previous_score && ot.all->training)
	  {
#ifdef DEBUG2
	    cout << "setting parent of feature index " << f.weight_index << endl;
#endif
	    
	    //	       cout << "parent by deviation " << log_delta << "\treg = " << entry[residual_regret] << "\t range = " << entry[residual_range] << "\t variance = " << entry[residual_variance] << "\t deviation = " << deviation << "\t delta_alpha_mult = " << delta_alpha_mult << endl;	
	    ot.best_previous_score = FLT_MAX;
	    entry->delta = log_delta;
	    entry->variance = 0.;
	    entry->range = 0.;
	    entry->value = 0.;
	    return true;
	  }
        else
	  return false;
      }
  }
  
  void update_regret(online_tree &ot, feature f, float base_loss, float feature_loss)
  {
    if(!ot.all->training)
      return;

    feature_stats* entry = get_entry(ot,f.weight_index);
    float diff = base_loss - feature_loss;

    entry->value += diff;
    entry->variance += diff*diff;
    if (entry->range < fabsf(diff))
      entry->range = fabsf(diff);

    //CHECK!!!!!

    if (entry->value < 0) {
	entry->value = 0;
        entry->variance = 0;
	entry->range = 0;
    }


    // cout << "DEBUG update_regret: Residual_variance " << entry[residual_variance] << endl;
    // cout << "DEBUG update_regret: Residual_range " << entry[residual_range] << endl;

  }
  
  void set_cycle(online_tree& ot, uint32_t weight_index) 
  {
    feature_stats* entry = get_entry(ot,weight_index);
    entry->parent |= 2;
  }
  
  bool used_feature(online_tree& ot, uint32_t weight_index)
  {
    feature_stats* entry = get_entry(ot,weight_index);
    return entry->parent & 2;
  }

  void create_inset_feature(online_tree& ot, example& ec, feature f);

  inline uint32_t get_weight_index(online_tree& ot, float& w) {
    uint32_t index = (uint32_t)(&w - ot.all->reg.weight_vector);
    uint32_t my_index;
    if (index == ot.f.weight_index) {
      uint64_t z = ot.f.weight_index;
      my_index =  (((size_t)(merand48(z) * ot.all->reg.weight_mask) << ot.all->reg.stride_shift) &
	      ot.all->reg.weight_mask);
    }
    else {
      my_index = (((index ^ ot.f.weight_index ) * mult_const ) & ot.all->reg.weight_mask);
    }
    return my_index;
  }
  
  inline void create_outset_feature(online_tree& ot, float v, float& w) {
    	  feature n;
          n.x = v * ot.f.x;
	  n.weight_index = get_weight_index(ot, w);
	  
          if (used_feature(ot, n.weight_index) || n.x == 0.0) return;

          set_cycle(ot, n.weight_index);

	  //check in_set here
          if (inset(ot, n, ot.derived_delta)) {
#ifdef DEBUG2
            cout << ec->example_counter << ": New feature " << n.weight_index << " = product of features " << ot.f.weight_index << " and " << index << endl;
#endif
	    feature temp_f = ot.f;
	    float temp_derived_delta = ot.derived_delta;

            create_inset_feature(ot, *(ot.original_ec), n);
	    ot.f = temp_f;
	    ot.derived_delta = temp_derived_delta;
          }
          else {
#ifdef DEBUG
            cout << "Putting a new feature " << n.weight_index << " in out set, product of features " << index << " and " << ot.f.weight_index << endl;
#endif
            ot.out_set.push_back(n);
          }

  }

  void set_outset(online_tree& ot, feature_stats* entry) 
  {
    entry->parent |= 1;
  }

  bool check_outset(online_tree& ot, feature_stats* entry)
  {
    return entry->parent & 1;
  }

  bool outset(online_tree& ot, feature_stats* entry, float w_val) {
    if(check_outset(ot, entry)) 
      return true;
    else {
      if(ot.best_previous_weight == FLT_MAX) return false;

      ot.best_weight_in_period = max(ot.best_weight_in_period, fabsf(w_val));
      float diff = fabsf(w_val - entry->value);
      entry->variance += diff*diff;
      entry->range = max(entry->range, diff);
      entry->value = w_val;     
 
      float deviation = get_deviation(entry->range, entry->variance, ot.alpha_mult, ot.derived_delta+log(ot.out_set.size()));
      if(fabsf(w_val) - deviation > ot.best_previous_weight) {
	//cout<<"Putting in outset "<<ot.synthetic.example_counter<<" w_val: "<<w_val<<" deviation: "<<deviation<<" prev_weight= "<<ot.best_previous_weight<<" current_best = "<<ot.best_weight_in_period<<endl;
	//cout<<"Called get_deviation with "<<entry->range<<" variance: "<<entry->variance<<" log_delta: "<<ot.derived_delta<<" "<<ot.alpha_mult<<endl;
	set_outset(ot, entry);
	//if(ot.out_set.size() > ot.all->length())
	  ot.best_previous_weight = FLT_MAX;
	return true;
      }
      else
	return false;
    }
  }
  
  void create_inset_feature(online_tree& ot, example& ec, feature f)
  {
  //cout << "DEBUG in create_new_features, feature index = " << f.weight_index << " value = " << f.x << endl;
    //    float* entry = get_entry(ot,f.weight_index);

    
    ot.synthetic.atomics[tree].push_back(f);
    ot.synthetic.num_features++;
    ot.synthetic.sum_feat_sq[tree] += f.x*f.x;

    float w_val = ot.all->reg.weight_vector[f.weight_index & ot.all->reg.weight_mask];
    feature_stats* entry = get_entry(ot, f.weight_index);    
 
    //check out_set bit here.
    if(outset(ot, entry, w_val)) {
      ot.current_depth++;
      if (ot.current_depth > ot.max_depth)
	ot.max_depth = ot.current_depth;
      
      ot.derived_delta = ot.derived_delta + 1 + log(ec.num_features);
      ot.f = f;
      GD::foreach_feature<online_tree, create_outset_feature>(*ot.all, ec, ot);
      ot.current_depth--;
    }
  }

  void setup_synthetic(online_tree& ot, example& ec)
  {//things to copy
    ot.synthetic.ld = ec.ld;
    ot.synthetic.tag = ec.tag;
    ot.synthetic.example_counter = ec.example_counter;
    ot.synthetic.ft_offset = ec.ft_offset;

    ot.synthetic.test_only = ec.test_only;
    ot.synthetic.end_pass = ec.end_pass;
    ot.synthetic.sorted = ec.sorted;
    ot.synthetic.in_use = ec.in_use;
    
    //things to set
    ot.synthetic.atomics[tree].erase();
    ot.synthetic.audit_features[tree].erase();
    ot.synthetic.num_features = 0;
    ot.synthetic.total_sum_feat_sq = 0;
    ot.synthetic.sum_feat_sq[tree] = 0;
    ot.synthetic.example_t = ec.example_t;

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

  
  inline void add_atomic(online_tree& ot, float v, float& w) 
  {
    uint32_t index = (uint32_t)(&w - ot.all->reg.weight_vector);
    set_cycle(ot, index);
    feature f = {v, index};
    create_inset_feature(ot, *(ot.original_ec), f);
  }
  
  void tree_features(online_tree& ot, example& ec)
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
    if (ec.example_counter==1) simple_print_features(*(ot.all), ec);
#endif
    
    ot.derived_delta = 1 + log(ec.num_features);
    ot.original_ec = &ec;
    GD::foreach_feature<online_tree, add_atomic>(*ot.all, ec, ot);
    ot.synthetic.total_sum_feat_sq = ot.synthetic.sum_feat_sq[tree];
    
    // cout << "DEBUG: Returning from tree_features, size of synthetic " << ot.synthetic.atomics[tree].size() << endl << endl;

    clear_features(ot, ot.synthetic.atomics[tree]);
    clear_features(ot, ot.out_set);
  }
  
  void learn(online_tree& ot, learner& base, example& ec)
  {
    if (ec.example_counter == ot.next)
      {
	ot.next = ot.next + ot.base_period * pow(ot.next / ot.base_period, ot.period_power);
	
	ot.best_previous_score = ot.best_score_in_period;
	ot.best_previous_weight = ot.best_weight_in_period;

	ot.best_score_in_period = 0.;
	ot.best_weight_in_period = 0.;
      }
    // cout << "---------------- new example, before entering tree features" << endl;
    tree_features(ot, ec);

    base.learn(ot.synthetic);
    ot.num_features = ot.synthetic.num_features;

    label_data* ld = (label_data*)(ec.ld);

    ec.final_prediction = ot.synthetic.final_prediction;
    ec.loss = ot.synthetic.loss;

    float base_loss = ot.synthetic.loss;

#ifdef DEBUG2
    cout << "example " << ot.synthetic.example_counter << " count " << ot.synthetic.atomics[tree].size();
    cout << " loss " << ot.synthetic.loss << "\t";
    simple_print_features(*(ot.all), &ot.synthetic);
#endif

    float old_base = ld->initial;
    ld->initial += ec.final_prediction;

    // cout << "final_prediction " << ec.final_prediction << endl;
    // cout << "loss " << ec.loss << endl;
    ot.synthetic.num_features = 1;

#ifdef DEBUG
    cout << "example " << ot.synthetic.example_counter <<
       " outset size "<< ot.out_set.size() << endl;
#endif
    
    for (feature *f = ot.out_set.begin; f != ot.out_set.end; f++)
      {
	ot.synthetic.atomics[tree].erase();
	ot.synthetic.atomics[tree].push_back(*f);

	ot.synthetic.sum_feat_sq[tree] = f->x*f->x;
	ot.synthetic.total_sum_feat_sq = ot.synthetic.sum_feat_sq[tree];

	// cout << "updating regret of feature " << f->weight_index << endl;
	base.learn(ot.synthetic);

	update_regret(ot, *f, base_loss, ot.synthetic.loss);
      }

    ld->initial = old_base;
  }

  void finish(online_tree& ot)
  {
    ot.synthetic.atomics[tree].delete_v();
    ot.out_set.delete_v();
    
    free(ot.per_feature);
  }

  void finish_online_tree_example(vw& all, online_tree& ot, example& ec)
  {
    size_t temp_num_features = ec.num_features;
    ec.num_features = ot.num_features;
    output_and_account_example(all,ec);
    ec.num_features = temp_num_features;
    VW::finish_example(all, &ec);
  }

  
  void save_load(online_tree& ot, io_buf& model_file, bool read, bool text)
  {
    cerr<<"save_load "<<endl;
    fflush(stderr);
    if (model_file.files.size() > 0) {
      size_t len = 0;
      for (size_t i = 0; i < ot.all->length(); i++)
	{ 
	  // char buff[512];
	  // uint32_t text_len = sprintf(buff, " parent: %ui residual_range: %f residual_variance: %f residual_regret: %f delta: %f\n", ot.per_feature[i].parent, ot.per_feature[i].range, ot.per_feature[i].variance, ot.per_feature[i].value, ot.per_feature[i].delta);
	  
	  len += bin_text_read_write_fixed(model_file, (char*)&ot.per_feature[i].parent, sizeof(ot.per_feature[i].parent),   "", read, "", 0, text);
	  len += bin_text_read_write_fixed(model_file, (char*)&ot.per_feature[i].range, sizeof(ot.per_feature[i].range),   "", read, "", 0, text);
	  len += bin_text_read_write_fixed(model_file, (char*)&ot.per_feature[i].variance, sizeof(ot.per_feature[i].variance),   "", read, "", 0, text);
	  len += bin_text_read_write_fixed(model_file, (char*)&ot.per_feature[i].value, sizeof(ot.per_feature[i].value),   "", read, "", 0, text);
	  len += bin_text_read_write_fixed(model_file, (char*)&ot.per_feature[i].delta, sizeof(ot.per_feature[i].delta),   "", read, "", 0, text);
	}
      cerr<<"Total r/w = "<<len<<endl;
    }
  }


  learner* setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file)
  {
    cerr<<"Starting setup\n";
    fflush(stderr);
    online_tree* data = (online_tree*)calloc(1,sizeof(online_tree));

    po::options_description desc("Online Tree options");
    desc.add_options()
      ("base_period", po::value<size_t>(), "number of examples between feature creation")
      ("period_power", po::value<float>(), "power on period");

    po::parsed_options parsed = po::command_line_parser(opts).
      style(po::command_line_style::default_style ^ po::command_line_style::allow_guessing).
      options(desc).allow_unregistered().run();
    opts = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    po::notify(vm);

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

    if (vm.count("base_period"))
      data->base_period = vm["base_period"].as<size_t>();
    else
      data->base_period = 100;

    data->next = data->base_period;

    if (vm.count("period_power"))
      data->period_power = vm["period_power"].as<float>();
    else
      data->period_power = 0.;

    data->per_feature = (feature_stats*)calloc(all.length(), sizeof(feature_stats));
    data->prev_variance = (float*)calloc(all.length(), sizeof(float));
    data->prev_value = (float*) calloc(all.length(), sizeof(float));

    data->best_previous_score = FLT_MAX;
    data->best_previous_weight = FLT_MAX;
    data->best_score_in_period = 0.;
    data->best_weight_in_period = 0.;
    
    for (size_t i = 0; i < all.length(); i++) {
      data->per_feature[i].parent = 0;
      data->per_feature[i].value = 0.;
      data->per_feature[i].range = 0.;
      data->per_feature[i].variance = 0.;
      data->per_feature[i].delta = 0.;      
      data->prev_variance[i] = 0.;
      data->prev_value[i] = 0.;
    }

    data->all = &all;
    learner* l = new learner(data, all.l);
    l->set_learn<online_tree, learn>();
    l->set_predict<online_tree, learn>();
    l->set_save_load<online_tree,save_load>();
    l->set_finish<online_tree,finish>();
    l->set_finish_example<online_tree,finish_online_tree_example>();
    l->set_end_pass<online_tree, allreduce_fstats>();
    cerr<<"Done with setup\n";
    fflush(stderr);
    return l;
  }
}
