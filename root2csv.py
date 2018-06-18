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
rng = TRandom3()

###############################

def RotateEvent( lep, jets, phi ):

    lep_new            = TLorentzVector( lep )
    lep_new.q          = lep.q
    lep_new.flav       = lep.flav    
    lep_new.topoetcone = lep.topoetcone
    lep_new.ptvarcone  = lep.ptvarcone
    lep_new.d0sig      = lep.d0sig
    
    lep_new.RotateZ( phi )

    jets_new = []
    for j in jets:
        jets_new += [ TLorentzVector(j) ]
        j_new = jets_new[-1]

        j_new.mv2c10 = j.mv2c10
        j_new.index  = j.index

        j_new.RotateZ( phi )

    return lep_new, jets_new

###############################

def MakeInput( jets, W_had, b_had, t_had, W_lep, b_lep, t_lep ):
   sjets = np.zeros( [ n_jets_per_event, n_features_per_jet ] )

   for i in range(len(jets)):
      jet = jets[i]
      sjets[i][0] = jet.Px()/GeV
      sjets[i][1] = jet.Py()/GeV 
      sjets[i][2] = jet.Pz()/GeV
      sjets[i][3] = jet.E()/GeV
      sjets[i][4] = jet.M()/GeV
      sjets[i][5] = jet.mv2c10
      
   target_W_had = np.zeros( [5] )
   target_b_had = np.zeros( [5] )
   target_t_had = np.zeros( [5] )
   target_W_lep = np.zeros( [5] )
   target_b_lep = np.zeros( [5] )  
   target_t_lep = np.zeros( [5] )
   
   target_t_had[0] = t_had.Px()/GeV
   target_t_had[1] = t_had.Py()/GeV
   target_t_had[2] = t_had.Pz()/GeV
   target_t_had[3] = t_had.E()/GeV
   target_t_had[4] = t_had.M()/GeV
    
   target_W_had[0] = W_had.Px()/GeV
   target_W_had[1] = W_had.Py()/GeV
   target_W_had[2] = W_had.Pz()/GeV
   target_W_had[3] = W_had.E()/GeV
   target_W_had[4] = W_had.M()/GeV

   target_b_had[0] = b_had.Px()/GeV
   target_b_had[1] = b_had.Py()/GeV
   target_b_had[2] = b_had.Pz()/GeV
   target_b_had[3] = b_had.E()/GeV
   target_b_had[4] = b_had.M()/GeV

   target_t_lep[0] = t_lep.Px()/GeV
   target_t_lep[1] = t_lep.Py()/GeV
   target_t_lep[2] = t_lep.Pz()/GeV
   target_t_lep[3] = t_lep.E()/GeV
   target_t_lep[4] = t_lep.M()/GeV
   
   target_W_lep[0] = W_lep.Px()/GeV
   target_W_lep[1] = W_lep.Py()/GeV
   target_W_lep[2] = W_lep.Pz()/GeV
   target_W_lep[3] = W_lep.E()/GeV
   target_W_lep[4] = W_lep.M()/GeV

   target_b_lep[0] = b_lep.Px()/GeV
   target_b_lep[1] = b_lep.Py()/GeV
   target_b_lep[2] = b_lep.Pz()/GeV
   target_b_lep[3] = b_lep.E()/GeV
   target_b_lep[4] = b_lep.M()/GeV

   return sjets, target_W_had, target_b_had, target_t_had, target_W_lep, target_b_lep, target_t_lep

###############################


filelistname = sys.argv[1]

syst = "nominal"
if len(sys.argv) > 2: syst = sys.argv[2]

outfilename = filelistname.split("/")[-1]
outfilename = "csv/topreco." + outfilename.replace(".txt", ".%s.csv" % ( syst ) )

# use data augmentation?
n_data_aug = 5
print "INFO: using data augmentation: rotateZ %ix" % n_data_aug

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

    el_n    = len(tree_reco.el_pt)
    mu_n    = len(tree_reco.mu_pt)
    lep_n   = el_n + mu_n 
    jets_n  = len(tree_reco.jet_pt)
    bjets_n = 0
    
    if lep_n > 1: continue
    if jets_n < 4: continue

    passed_ejets  = False
    passed_mujets = False
    if   (el_n == 1) and (mu_n == 0): passed_ejets = True
    elif (el_n == 0) and (mu_n == 1): passed_mujets = True
    else: continue
    
    #passed_ejets  = tree_reco.passed_resolved_ejets_4j2b_2015 or tree_reco.passed_resolved_ejets_4j2b_2016
    #passed_mujets = tree_reco.passed_resolved_mujets_4j2b_2015 or tree_reco.passed_resolved_mujets_4j2b_2016
    #accepted = passed_ejets or passed_mujets
    #if not accepted: continue

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
    
    jets = []
    for i in range(jets_n):
        if i >= n_jets_per_event: break
        
        jets += [ TLorentzVector() ]
        j = jets[-1]
        j.index = i
        j.SetPtEtaPhiE( tree_reco.jet_pt[i], tree_reco.jet_eta[i], tree_reco.jet_phi[i], tree_reco.jet_e[i] )
        j.mv2c10 = tree_reco.jet_mv2c10[i]
        if j.mv2c10 > 0.83: bjets_n += 1

    jets.sort( key=lambda jet: jet.mv2c10, reverse=True )
        
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

    W_from_t = TLorentzVector() 
    W_from_t.SetPtEtaPhiM( tree_parton.MC_W_from_t_pt,
                           tree_parton.MC_W_from_t_eta,
                           tree_parton.MC_W_from_t_phi,
                           tree_parton.MC_W_from_t_m )

    W_from_tb = TLorentzVector()
    W_from_tb.SetPtEtaPhiM( tree_parton.MC_W_from_tbar_pt,
                            tree_parton.MC_W_from_tbar_eta,
                            tree_parton.MC_W_from_tbar_phi,
                            tree_parton.MC_W_from_tbar_m )

    b_from_t = TLorentzVector() 
    b_from_t.SetPtEtaPhiM( tree_parton.MC_b_from_t_pt,
                           tree_parton.MC_b_from_t_eta,
                           tree_parton.MC_b_from_t_phi,
                           tree_parton.MC_b_from_t_m )

    b_from_tb = TLorentzVector()
    b_from_tb.SetPtEtaPhiM( tree_parton.MC_b_from_tbar_pt,
                            tree_parton.MC_b_from_tbar_eta,
                            tree_parton.MC_b_from_tbar_phi,
                            tree_parton.MC_b_from_tbar_m )


    if t.Pt() == 0.: continue
    if tb.Pt() == 0.: continue
    if t.M() != t.M(): continue
    if tb.M() != tb.M(): continue

    # determine hadronic and leptonic top
    pid_Wdecay1_from_t    = tree_parton.MC_Wdecay2_from_t_pdgId
    pid_Wdecay1_from_tbar = tree_parton.MC_Wdecay2_from_tbar_pdgId
    #print "DEBUG:", pid_Wdecay1_from_t, pid_Wdecay1_from_tbar
    

    decay_channel = -1
    apid_Wdecay1_from_t  = abs( tree_parton.MC_Wdecay1_from_t_pdgId )
    apid_Wdecay1_from_tb = abs( tree_parton.MC_Wdecay1_from_tbar_pdgId )

    if   (apid_Wdecay1_from_t < 10) and (apid_Wdecay1_from_tb < 10): decay_channel = 0
    elif (apid_Wdecay1_from_t < 10) and (apid_Wdecay1_from_tb > 10): decay_channel = 1
    elif (apid_Wdecay1_from_t > 10) and (apid_Wdecay1_from_tb < 10): decay_channel = 1
    else:                                                            decay_channel = 2

    if not decay_channel == 1: continue

    t_had = None
    t_lep = None
    W_had = None
    W_lep = None
    b_had = None
    b_lep = None

    if apid_Wdecay1_from_t < 10:
       # t->t_had, tb->t_lep
       
       t_had = TLorentzVector(t)
       W_had = TLorentzVector(W_from_t)
       b_had = TLorentzVector(b_from_t)
       
       t_lep = TLorentzVector(tb)
       W_lep = TLorentzVector(W_from_tb)
       b_lep = TLorentzVector(b_from_tb)

    else:
       # t->t_lep, tb->t_had
       
       t_had = TLorentzVector(tb)
       W_had = TLorentzVector(W_from_tb)
       b_had = TLorentzVector(b_from_tb)
       
       t_lep = TLorentzVector(t)
       W_lep = TLorentzVector(W_from_t)
       b_lep = TLorentzVector(b_from_t)
       
    if t_had == None or t_lep == None:
      continue

    n_good += 1
   
    phi = 0.
    for n in range(n_data_aug+1):
       # rotate f.s.o.
       lep.RotateZ( phi )
       met_phi += phi
       for j in jets: j.RotateZ(phi)
       W_had.RotateZ( phi )
       b_had.RotateZ( phi )
       t_had.RotateZ( phi )
       W_lep.RotateZ( phi )
       b_lep.RotateZ( phi )
       t_lep.RotateZ( phi )
       
       # make event wrapper
       sjets, target_W_had, target_b_had, target_t_had, target_W_lep, target_b_lep, target_t_lep = MakeInput( jets, W_had, b_had, t_had, W_lep, b_lep, t_lep )
   
       # write out
       csvwriter.writerow( (
          "%i" % tree_reco.runNumber, "%i" % tree_reco.eventNumber, "%.3f" % weight, "%i" % jets_n, "%i" % bjets_n,
          "%.3f" % lep.Px(),     "%.3f" % lep.Py(),     "%.3f" % lep.Pz(),     "%.3f" % lep.E(),      "%.3f" % met_met,      "%.3f" % met_phi,
          "%.3f" % sjets[0][0],  "%.3f" % sjets[0][1],  "%.3f" % sjets[0][2],  "%.3f" % sjets[0][3],  "%.3f" % sjets[0][4],  "%.3f" % sjets[0][5], 
          "%.3f" % sjets[1][0],  "%.3f" % sjets[1][1],  "%.3f" % sjets[1][2],  "%.3f" % sjets[1][3],  "%.3f" % sjets[1][4],  "%.3f" % sjets[1][5], 
          "%.3f" % sjets[2][0],  "%.3f" % sjets[2][1],  "%.3f" % sjets[2][2],  "%.3f" % sjets[2][3],  "%.3f" % sjets[2][4],  "%.3f" % sjets[2][5],
          "%.3f" % sjets[3][0],  "%.3f" % sjets[3][1],  "%.3f" % sjets[3][2],  "%.3f" % sjets[3][3],  "%.3f" % sjets[3][4],  "%.3f" % sjets[3][5], 
          "%.3f" % sjets[4][0],  "%.3f" % sjets[4][1],  "%.3f" % sjets[4][2],  "%.3f" % sjets[4][3],  "%.3f" % sjets[4][4],  "%.3f" % sjets[4][5], 
          "%.3f" % target_W_had[0], "%.3f" % target_W_had[1], "%.3f" % target_W_had[2], "%.3f" % target_W_had[3], "%.3f" % target_W_had[4],
          "%.3f" % target_W_lep[0], "%.3f" % target_W_lep[1], "%.3f" % target_W_lep[2], "%.3f" % target_W_lep[3], "%.3f" % target_W_lep[4],
          "%.3f" % target_b_had[0], "%.3f" % target_b_had[1], "%.3f" % target_b_had[2], "%.3f" % target_b_had[3], "%.3f" % target_b_had[4],
          "%.3f" % target_b_lep[0], "%.3f" % target_b_lep[1], "%.3f" % target_b_lep[2], "%.3f" % target_b_lep[3], "%.3f" % target_b_lep[4],
          "%.3f" % target_t_had[0], "%.3f" % target_t_had[1], "%.3f" % target_t_had[2], "%.3f" % target_t_had[3], "%.3f" % target_t_had[4],
          "%.3f" % target_t_lep[0], "%.3f" % target_t_lep[1], "%.3f" % target_t_lep[2], "%.3f" % target_t_lep[3], "%.3f" % target_t_lep[4]
       ) )

       phi = rng.Uniform( -TMath.Pi(), TMath.Pi() )

    
        
outfile.close()

f_good = 100. * n_good / n_entries_reco
print "INFO: %i entries written (%.2f %%)" % ( n_good, f_good) 
