event based risk
================

============================================== ========================
gem-tstation:/home/michele/ssd/calc_60077.hdf5 Tue Oct 11 06:56:09 2016
engine_version                                 2.1.0-git4e31fdd        
hazardlib_version                              0.21.0-gitab31f47       
============================================== ========================

num_sites = 7, sitecol = 1015 B

Parameters
----------
============================ ================================
calculation_mode             'event_based_risk'              
number_of_logic_tree_samples 0                               
maximum_distance             {u'Active Shallow Crust': 200.0}
investigation_time           10000.0                         
ses_per_logic_tree_path      1                               
truncation_level             3.0                             
rupture_mesh_spacing         2.0                             
complex_fault_mesh_spacing   2.0                             
width_of_mfd_bin             0.1                             
area_source_discretization   None                            
random_seed                  24                              
master_seed                  42                              
avg_losses                   False                           
============================ ================================

Input files
-----------
======================= ============================================================
Name                    File                                                        
======================= ============================================================
exposure                `exposure_model.xml <exposure_model.xml>`_                  
gsim_logic_tree         `gsim_logic_tree.xml <gsim_logic_tree.xml>`_                
job_ini                 `job.ini <job.ini>`_                                        
occupants_vulnerability `vulnerability_model.xml <vulnerability_model.xml>`_        
source                  `source_model.xml <source_model.xml>`_                      
source_model_logic_tree `source_model_logic_tree.xml <source_model_logic_tree.xml>`_
======================= ============================================================

Composite source model
----------------------
========= ====== ====================================== =============== ================
smlt_path weight source_model_file                      gsim_logic_tree num_realizations
========= ====== ====================================== =============== ================
b1        1.000  `source_model.xml <source_model.xml>`_ trivial(1)      1/1             
========= ====== ====================================== =============== ================

Required parameters per tectonic region type
--------------------------------------------
====== =================== ========= ========== ==========
grp_id gsims               distances siteparams ruptparams
====== =================== ========= ========== ==========
0      BooreAtkinson2008() rjb       vs30       rake mag  
====== =================== ========= ========== ==========

Realizations per (TRT, GSIM)
----------------------------

::

  <RlzsAssoc(size=1, rlzs=1)
  0,BooreAtkinson2008(): ['<0,b1~b1,w=1.0>']>

Number of ruptures per tectonic region type
-------------------------------------------
================ ====== ==================== =========== ============ ============
source_model     grp_id trt                  num_sources eff_ruptures tot_ruptures
================ ====== ==================== =========== ============ ============
source_model.xml 0      Active Shallow Crust 1           265          482         
================ ====== ==================== =========== ============ ============

Informational data
------------------
====================================== ============
compute_ruptures_max_received_per_task 89,383      
compute_ruptures_num_tasks             3           
compute_ruptures_sent.gsims            279         
compute_ruptures_sent.monitor          2,886       
compute_ruptures_sent.sitecol          2,235       
compute_ruptures_sent.sources          8,233       
compute_ruptures_tot_received          249,523     
hazard.input_weight                    482         
hazard.n_imts                          1           
hazard.n_levels                        1           
hazard.n_realizations                  1           
hazard.n_sites                         7           
hazard.n_sources                       1           
hazard.output_weight                   7.000       
hostname                               gem-tstation
require_epsilons                       1           
====================================== ============

Estimated data transfer for the avglosses
-----------------------------------------
7 asset(s) x 1 realization(s) x 1 loss type(s) x 1 losses x 8 bytes x 50 tasks = 2.73 KB

Exposure model
--------------
=============== ========
#assets         7       
#taxonomies     1       
deductibile     absolute
insurance_limit absolute
=============== ========

======== ===== ====== === === ========= ==========
taxonomy mean  stddev min max num_sites num_assets
tax1     1.000 0.0    1   1   7         7         
======== ===== ====== === === ========= ==========

Slowest sources
---------------
====== ========= ================= ============ ========= ========= =========
grp_id source_id source_class      num_ruptures calc_time num_sites num_split
====== ========= ================= ============ ========= ========= =========
0      1         SimpleFaultSource 482          0.0       7         0        
====== ========= ================= ============ ========= ========= =========

Computation times by source typology
------------------------------------
================= ========= ======
source_class      calc_time counts
================= ========= ======
SimpleFaultSource 0.0       1     
================= ========= ======

Information about the tasks
---------------------------
================== ===== ====== ===== ===== =========
operation-duration mean  stddev min   max   num_tasks
compute_ruptures   0.209 0.050  0.154 0.253 3        
================== ===== ====== ===== ===== =========

Slowest operations
------------------
================================ ========= ========= ======
operation                        time_sec  memory_mb counts
================================ ========= ========= ======
total compute_ruptures           0.626     3.371     3     
filtering ruptures               0.474     0.0       265   
managing sources                 0.087     0.0       1     
split/filter heavy sources       0.085     0.0       1     
saving ruptures                  0.010     0.0       3     
reading composite source model   0.009     0.0       1     
reading exposure                 0.005     0.0       1     
filtering composite source model 0.003     0.0       1     
store source_info                8.502E-04 0.0       1     
Initializing rupture serials     4.292E-05 0.0       1     
reading site collection          8.106E-06 0.0       1     
================================ ========= ========= ======