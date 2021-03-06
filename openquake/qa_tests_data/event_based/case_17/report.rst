Event Based Hazard QA Test, Case 17
===================================

============================================== ========================
gem-tstation:/home/michele/ssd/calc_60108.hdf5 Tue Oct 11 06:57:55 2016
engine_version                                 2.1.0-git4e31fdd        
hazardlib_version                              0.21.0-gitab31f47       
============================================== ========================

num_sites = 1, sitecol = 739 B

Parameters
----------
============================ ================================
calculation_mode             'event_based'                   
number_of_logic_tree_samples 5                               
maximum_distance             {u'active shallow crust': 200.0}
investigation_time           1.0                             
ses_per_logic_tree_path      3                               
truncation_level             2.0                             
rupture_mesh_spacing         1.0                             
complex_fault_mesh_spacing   1.0                             
width_of_mfd_bin             1.0                             
area_source_discretization   10.0                            
random_seed                  106                             
master_seed                  0                               
============================ ================================

Input files
-----------
======================= ============================================================
Name                    File                                                        
======================= ============================================================
gsim_logic_tree         `gsim_logic_tree.xml <gsim_logic_tree.xml>`_                
job_ini                 `job.ini <job.ini>`_                                        
source                  `source_model_1.xml <source_model_1.xml>`_                  
source                  `source_model_2.xml <source_model_2.xml>`_                  
source_model_logic_tree `source_model_logic_tree.xml <source_model_logic_tree.xml>`_
======================= ============================================================

Composite source model
----------------------
========= ====== ========================================== =============== ================
smlt_path weight source_model_file                          gsim_logic_tree num_realizations
========= ====== ========================================== =============== ================
b1        0.200  `source_model_1.xml <source_model_1.xml>`_ trivial(1)      1/0             
b2        0.200  `source_model_2.xml <source_model_2.xml>`_ trivial(1)      4/1             
========= ====== ========================================== =============== ================

Required parameters per tectonic region type
--------------------------------------------
====== ================ ========= ========== ==========
grp_id gsims            distances siteparams ruptparams
====== ================ ========= ========== ==========
1      SadighEtAl1997() rrup      vs30       rake mag  
====== ================ ========= ========== ==========

Realizations per (TRT, GSIM)
----------------------------

::

  <RlzsAssoc(size=1, rlzs=5)
  1,SadighEtAl1997(): ['<1,b2~b1,w=0.2>', '<2,b2~b1,w=0.2>', '<3,b2~b1,w=0.2>', '<4,b2~b1,w=0.2>']>

Number of ruptures per tectonic region type
-------------------------------------------
================== ====== ==================== =========== ============ ============
source_model       grp_id trt                  num_sources eff_ruptures tot_ruptures
================== ====== ==================== =========== ============ ============
source_model_2.xml 1      Active Shallow Crust 1           3            7           
================== ====== ==================== =========== ============ ============

Informational data
------------------
====================================== ============
compute_ruptures_max_received_per_task 5,211       
compute_ruptures_num_tasks             2           
compute_ruptures_sent.gsims            164         
compute_ruptures_sent.monitor          2,032       
compute_ruptures_sent.sitecol          866         
compute_ruptures_sent.sources          3,141       
compute_ruptures_tot_received          6,915       
hazard.input_weight                    6.700       
hazard.n_imts                          1           
hazard.n_levels                        3           
hazard.n_realizations                  5           
hazard.n_sites                         1           
hazard.n_sources                       2           
hazard.output_weight                   0.150       
hostname                               gem-tstation
====================================== ============

Specific information for event based
------------------------------------
======================== =====
Total number of ruptures 3    
Total number of events   1    
Rupture multiplicity     0.333
======================== =====

Slowest sources
---------------
====== ========= ============ ============ ========= ========= =========
grp_id source_id source_class num_ruptures calc_time num_sites num_split
====== ========= ============ ============ ========= ========= =========
1      2         PointSource  7            0.0       1         0        
0      1         PointSource  39           0.0       1         0        
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
================== ===== ====== ===== ===== =========
operation-duration mean  stddev min   max   num_tasks
compute_ruptures   0.016 0.014  0.006 0.026 2        
================== ===== ====== ===== ===== =========

Slowest operations
------------------
================================ ========= ========= ======
operation                        time_sec  memory_mb counts
================================ ========= ========= ======
total compute_ruptures           0.032     0.0       2     
reading composite source model   0.008     0.0       1     
saving ruptures                  0.006     0.0       2     
filtering composite source model 0.003     0.0       1     
managing sources                 0.003     0.0       1     
filtering ruptures               0.002     0.0       3     
store source_info                8.180E-04 0.0       1     
Initializing rupture serials     4.911E-05 0.0       1     
reading site collection          4.482E-05 0.0       1     
================================ ========= ========= ======