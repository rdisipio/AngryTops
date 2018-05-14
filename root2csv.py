#!/usr/bin/env python

import os, sys
import csv
from math import log, exp, sqrt

from ROOT import *
import numpy as np

gROOT.SetBatch(True)

from features import *

###############################

GeV = 1e3
TeV = 1e6

###############################

filelistname = sys.argv[1]

syst = "nominal"
if len(sys.argv) > 2: syst = sys.argv[2]

outfilename = filelistname.split("/")[-1]
outfilename = "csv/topreco." + outfilename.replace(".txt", ".%s.csv" % ( syst ) )

outfile = open( outfilename, "wt" )
csvwriter = csv.writer( outfile )

print "INFO: output file:", outfilename

treename = "nominal"

tree_reco   = TChain( treename, treename )
tree_parton = TChain( "truth", "truth" )
f = open( filelistname, 'r' )
for fname in f.readlines():
   fname = fname.strip()
#   print "DEBUG: adding file:", fname
   tree_reco.AddFile( fname )
   tree_parton.AddFile( fname )

n_entries_reco = tree_reco.GetEntries()
n_entries_parton = tree_parton.GetEntries()
print "INFO: reco   entries found:", n_entries_reco
print "INFO: parton entries found:", n_entries_parton

success = tree_parton.BuildIndex( "runNumber", "eventNumber" )

n_good = 0
for ientry in range(n_entries_reco):
    tree_reco.GetEntry( ientry )

    if ( n_entries_reco < 10 ) or ( (ientry+1) % int(float(n_entries_reco)/10.)  == 0 ):
        perc = 100. * ientry / float(n_entries_reco)
        print "INFO: Event %-9i  (%3.0f %%)" % ( ientry, perc )

    passed_ejets  = tree_reco.passed_resolved_ejets_4j2b_2015 or tree_reco.passed_resolved_ejets_4j2b_2016
    passed_mujets = tree_reco.passed_resolved_mujets_4j2b_2015 or tree_reco.passed_resolved_mujets_4j2b_2016
    accepted = passed_ejets or passed_mujets
    if not accepted: continue

    mcChannelNumber = tree_reco.mcChannelNumber
    runNumber       = tree_reco.runNumber
    eventNumber     = tree_reco.eventNumber
    weight          = 1.0

    ientry_parton = tree_parton.GetEntryNumberWithIndex( runNumber, eventNumber )
    tree_parton.GetEntry( ientry_parton )

    lep = TLorentzVector()
    if passed_ejets:
       lep.SetPtEtaPhiE( tree_reco.el_pt[0]/GeV, tree_reco.el_eta[0], tree_reco.el_phi[0], tree_reco.el_e[0]/GeV )
    else:
       lep.SetPtEtaPhiE( tree_reco.mu_pt[0]/GeV, tree_reco.mu_eta[0], tree_reco.mu_phi[0], tree_reco.mu_e[0]/GeV )

    met_met = tree_reco.met_met/GeV
    met_phi = tree_reco.met_phi
    
    jets_n = len(tree_reco.jet_pt)
    jets = []
    for i in range(jets_n):
        if i >= n_jets_per_event: break
        
        jets += [ TLorentzVector() ]
        j = jets[-1]
        j.index = i
        j.SetPtEtaPhiE( tree_reco.jet_pt[i], tree_reco.jet_eta[i], tree_reco.jet_phi[i], tree_reco.jet_e[i] )
        j.mv2c10 = tree_reco.jet_mv2c10[i]
        
    t = TLorentzVector()
    t.SetPtEtaPhiM( tree_parton.MC_t_afterFSR_pt,
                    tree_parton.MC_t_afterFSR_eta,
                    tree_parton.MC_t_afterFSR_phi,
                    tree_parton.MC_t_afterFSR_m )
    
    tb =  TLorentzVector()
    tb.SetPtEtaPhiM( tree_parton.MC_tbar_afterFSR_pt,
                       tree_parton.MC_tbar_afterFSR_eta,
                       tree_parton.MC_tbar_afterFSR_phi,
                       tree_parton.MC_tbar_afterFSR_m )

    if t.Pt() == 0.: continue
    if tb.Pt() == 0.: continue
    if t.M() != t.M(): continue
    if tb.M() != tb.M(): continue

    # determine hadronic and leptonic top
    pid_Wdecay1_from_t    = tree_parton.MC_Wdecay2_from_t_pdgId
    pid_Wdecay1_from_tbar = tree_parton.MC_Wdecay2_from_tbar_pdgId
    #print "DEBUG:", pid_Wdecay1_from_t, pid_Wdecay1_from_tbar
    
    t_had = None
    t_lep = None
    if abs(pid_Wdecay1_from_t) < 10:
       t_had = TLorentzVector(t)
    else:
       t_lep = TLorentzVector(t)

    if abs(pid_Wdecay1_from_tbar) < 10:
       t_had = TLorentzVector(tb)
    else:
       t_lep = TLorentzVector(tb)
       
    if t_had == None or t_lep == None:
      continue
    
    # make event wrapper
    sjets = np.zeros( [ n_jets_per_event, n_features_per_jet ] )

    for i in range(len(jets)):
        jet = jets[i]
        sjets[i][0] = jet.Px()/GeV
        sjets[i][1] = jet.Py()/GeV 
        sjets[i][2] = jet.Pz()/GeV
        sjets[i][3] = jet.E()/GeV
        sjets[i][4] = jet.M()/GeV
        sjets[i][5] = jet.mv2c10

    target_had = np.zeros( [5] )
    target_lep = np.zeros( [5] )
     
    target_had[0] = t_had.Px()/GeV
    target_had[1] = t_had.Py()/GeV
    target_had[2] = t_had.Pz()/GeV
    target_had[3] = t_had.E()/GeV
    target_had[4] = t_had.M()/GeV
    
    target_lep[0] = t_lep.Px()/GeV
    target_lep[1] = t_lep.Py()/GeV
    target_lep[2] = t_lep.Pz()/GeV
    target_lep[3] = t_lep.E()/GeV
    target_lep[4] = t_lep.M()/GeV

    # write out
    csvwriter.writerow( (
       "%i" % tree_reco.runNumber, "%i" % tree_reco.eventNumber, "%.3f" % weight,
       "%.3f" % lep.Px(),     "%.3f" % lep.Py(),     "%.3f" % lep.Pz(),     "%.3f" % lep.E(),      "%.3f" % met_met,      "%.3f" % met_phi,
       "%.3f" % sjets[0][0],  "%.3f" % sjets[0][1],  "%.3f" % sjets[0][2],  "%.3f" % sjets[0][3],  "%.3f" % sjets[0][4],  "%.3f" % sjets[0][5], 
       "%.3f" % sjets[1][0],  "%.3f" % sjets[1][1],  "%.3f" % sjets[1][2],  "%.3f" % sjets[1][3],  "%.3f" % sjets[1][4],  "%.3f" % sjets[1][5], 
       "%.3f" % sjets[2][0],  "%.3f" % sjets[2][1],  "%.3f" % sjets[2][2],  "%.3f" % sjets[2][3],  "%.3f" % sjets[2][4],  "%.3f" % sjets[2][5],
       "%.3f" % sjets[3][0],  "%.3f" % sjets[3][1],  "%.3f" % sjets[3][2],  "%.3f" % sjets[3][3],  "%.3f" % sjets[3][4],  "%.3f" % sjets[3][5], 
       "%.3f" % sjets[4][0],  "%.3f" % sjets[4][1],  "%.3f" % sjets[4][2],  "%.3f" % sjets[4][3],  "%.3f" % sjets[4][4],  "%.3f" % sjets[4][5], 
       "%.3f" % target_had[0], "%.3f" % target_had[1], "%.3f" % target_had[2], "%.3f" % target_had[3], "%.3f" % target_had[4],
       "%.3f" % target_lep[0], "%.3f" % target_lep[1], "%.3f" % target_lep[2], "%.3f" % target_lep[3], "%.3f" % target_lep[4]
    ) )

    n_good += 1
        
outfile.close()

f_good = 100. * n_good / n_entries_reco
print "INFO: %i entries written (%.2f %%)" % ( n_good, f_good) 
