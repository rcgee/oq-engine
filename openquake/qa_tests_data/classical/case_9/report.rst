Classical Hazard QA Test, Case 9
================================

============================================== ========================
gem-tstation:/home/michele/ssd/calc_60090.hdf5 Tue Oct 11 06:56:48 2016
engine_version                                 2.1.0-git4e31fdd        
hazardlib_version                              0.21.0-gitab31f47       
============================================== ========================

num_sites = 1, sitecol = 739 B

Parameters
----------
============================ ================================
calculation_mode             'classical'                     
number_of_logic_tree_samples 0                               
maximum_distance             {u'active shallow crust': 200.0}
investigation_time           1.0                             
ses_per_logic_tree_path      1                               
truncation_level             0.0                             
rupture_mesh_spacing         0.01                            
complex_fault_mesh_spacing   0.01                            
width_of_mfd_bin             0.001                           
area_source_discretization   10.0                            
random_seed                  1066                            
master_seed                  0                               
sites_per_tile               10000                           
============================ ================================

Input files
-----------
======================= ============================================================
Name                    File                                                        
======================= ============================================================
gsim_logic_tree         `gsim_logic_tree.xml <gsim_logic_tree.xml>`_                
job_ini                 `job.ini <job.ini>`_                                        
source                  `7.0 <7.0>`_                                                
source                  `7.5 <7.5>`_                                                
source                  `source_model.xml <source_model.xml>`_                      
source_model_logic_tree `source_model_logic_tree.xml <source_model_logic_tree.xml>`_
======================= ============================================================

Composite source model
----------------------
========= ====== ====================================== =============== ================
smlt_path weight source_model_file                      gsim_logic_tree num_realizations
========= ====== ====================================== =============== ================
b1_b2     0.500  `source_model.xml <source_model.xml>`_ trivial(1)      1/1             
b1_b3     0.500  `source_model.xml <source_model.xml>`_ trivial(1)      1/1             
========= ====== ====================================== =============== ================

Required parameters per tectonic region type
--------------------------------------------
====== ================ ========= ========== ==========
grp_id gsims            distances siteparams ruptparams
====== ================ ========= ========== ==========
0      SadighEtAl1997() rrup      vs30       rake mag  
1      SadighEtAl1997() rrup      vs30       rake mag  
====== ================ ========= ========== ==========

Realizations per (TRT, GSIM)
----------------------------

::

  <RlzsAssoc(size=2, rlzs=2)
  0,SadighEtAl1997(): ['<0,b1_b2~b1,w=0.5>']
  1,SadighEtAl1997(): ['<1,b1_b3~b1,w=0.5>']>

Number of ruptures per tectonic region type
-------------------------------------------
================ ====== ==================== =========== ============ ============
source_model     grp_id trt                  num_sources eff_ruptures tot_ruptures
================ ====== ==================== =========== ============ ============
source_model.xml 0      Active Shallow Crust 1           3000         3,000       
source_model.xml 1      Active Shallow Crust 1           3500         3,500       
================ ====== ==================== =========== ============ ============

============= =====
#TRT models   2    
#sources      2    
#eff_ruptures 6,500
#tot_ruptures 6,500
#tot_weight   650  
============= =====

Informational data
------------------
======================================== ============
count_eff_ruptures_max_received_per_task 1,244       
count_eff_ruptures_num_tasks             2           
count_eff_ruptures_sent.gsims            164         
count_eff_ruptures_sent.monitor          2,052       
count_eff_ruptures_sent.sitecol          1,154       
count_eff_ruptures_sent.sources          2,428       
count_eff_ruptures_tot_received          2,488       
hazard.input_weight                      650         
hazard.n_imts                            1           
hazard.n_levels                          4           
hazard.n_realizations                    2           
hazard.n_sites                           1           
hazard.n_sources                         2           
hazard.output_weight                     8.000       
hostname                                 gem-tstation
======================================== ============

Slowest sources
---------------
====== ========= ============ ============ ========= ========= =========
grp_id source_id source_class num_ruptures calc_time num_sites num_split
====== ========= ============ ============ ========= ========= =========
1      1         PointSource  3,500        0.0       1         0        
0      1         PointSource  3,000        0.0       1         0        
====== ========= ============ ============ ========= ========= =========

Computation times by source typology
------------------------------------
============ ========= ======
source_class calc_time counts
============ ========= ======
PointSource  0.0       2     
============ ========= ======

Information about the tasks
---------------------------
================== ========= ========= ========= ========= =========
operation-duration mean      stddev    min       max       num_tasks
count_eff_ruptures 7.241E-04 8.429E-06 7.181E-04 7.300E-04 2        
================== ========= ========= ========= ========= =========

Slowest operations
------------------
================================ ========= ========= ======
operation                        time_sec  memory_mb counts
================================ ========= ========= ======
reading composite source model   0.025     0.0       1     
filtering composite source model 0.019     0.0       1     
managing sources                 0.009     0.0       1     
split/filter heavy sources       0.006     0.0       2     
total count_eff_ruptures         0.001     0.0       2     
store source_info                7.758E-04 0.0       1     
aggregate curves                 5.794E-05 0.0       2     
reading site collection          3.910E-05 0.0       1     
saving probability maps          3.481E-05 0.0       1     
================================ ========= ========= ======