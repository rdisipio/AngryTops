header = [
    "runNumber", "eventNumber", "weight",
    "lep_px", "lep_py", "lep_pz", "lep_E", "met_met", "met_phi",
    "j1_px", "j1_py", "j1_pz", "j1_E", "j1_m", "j1_mv2c10",
    "j2_px", "j2_py", "j2_pz", "j2_E", "j2_m", "j2_mv2c10",
    "j3_px", "j3_py", "j3_pz", "j3_E", "j3_m", "j3_mv2c10",
    "j4_px", "j4_py", "j4_pz", "j4_E", "j4_m", "j4_mv2c10",
    "j5_px", "j5_py", "j5_pz", "j5_E", "j5_m", "j5_mv2c10",
    "t_had_px", "t_had_py", "t_had_pz", "t_had_E", "t_had_m", 
    "t_lep_px", "t_lep_py", "t_lep_pz", "t_lep_E", "t_lep_m", 
]

input_features = [
    "lep_px", "lep_py", "lep_pz", "lep_E", "met_met", "met_phi",
    "j1_px", "j1_py", "j1_pz", "j1_E", "j1_m", "j1_mv2c10",
    "j2_px", "j2_py", "j2_pz", "j2_E", "j2_m", "j2_mv2c10",
    "j3_px", "j3_py", "j3_pz", "j3_E", "j3_m", "j3_mv2c10",
    "j4_px", "j4_py", "j4_pz", "j4_E", "j4_m", "j4_mv2c10",
    "j5_px", "j5_py", "j5_pz", "j5_E", "j5_m", "j5_mv2c10",
]

target_features_ttbar = [
  "t_had_px", "t_had_py", "t_had_pz", "t_had_E", 
  "t_lep_px", "t_lep_py", "t_lep_pz", "t_lep_E",
]

target_features_t_had = [
  "t_had_px", "t_had_py", "t_had_pz", "t_had_E",
]

target_feature_t_lep = [
  "t_lep_px", "t_lep_py", "t_lep_pz", "t_lep_E",
]

n_jets_per_event   = 5
n_features_per_jet = 6 # (px, py, pz, E, M, bw )
n_features_per_top = len(target_features_t_had) # (px, py, pz, E )


