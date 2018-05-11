#!/usr/bin/env python

import os, sys
import csv
from math import log, exp, sqrt

from ROOT import *
import numpy as np

gROOT.SetBatch(True)

###############################

GeV = 1e3
TeV = 1e6

###############################

n_jets_max = 7
n_features_per_jet = 6 # (px, py, pz, E, M, bw )

###############################


filelistname = sys.argv[1]

syst = "nominal"
if len(sys.argv) > 2: syst = sys.argv[2]

outfilename = filelistname.split("/")[-1]
outfilename = "csv/topreco_" + outfilename.replace(".txt", ".%s.csv" % ( syst ) )

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

for ientry in range(n_entries_reco):
    tree_reco.GetEntry( ientry )

    if ( n_entries_reco < 10 ) or ( (ientry+1) % int(float(n_entries_reco)/10.)  == 0 ):
        perc = 100. * ientry / float(n_entries_reco)
        print "INFO: Event %-9i  (%3.0f %%)" % ( ientry, perc )

    mcChannelNumber = tree_reco.mcChannelNumber
    runNumber       = tree_reco.runNumber
    eventNumber     = tree_reco.eventNumber

    weight = 1.0

    ientry_parton = tree_parton.GetEntryNumberWithIndex( runNumber, eventNumber )
    tree_parton.GetEntry( ientry_parton )

    jets_n = len(tree_reco.jet_pt)
    jets = []
    for i in range(jets_n):
        if i >= n_jets_max: break
        
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
    
    tbar =  TLorentzVector()
    tbar.SetPtEtaPhiM( tree_parton.MC_tbar_afterFSR_pt,
                       tree_parton.MC_tbar_afterFSR_eta,
                       tree_parton.MC_tbar_afterFSR_phi,
                       tree_parton.MC_tbar_afterFSR_m )

    # make event wrapper
    event = np.zeros( [ n_jets_max, n_features_per_jet ] )

    for i in range(len(jets)):
        jet = jets[i]
        event[i][0] = jet.Px()/GeV
        event[i][1] = jet.Py()/GeV 
        event[i][2] = jet.Pz()/GeV
        event[i][3] = jet.E()/GeV
        event[i][4] = jet.M()/GeV
        event[i][5] = jet.mv2c10

    # write out
    csvwriter.writerow( (
        "%i" % tree_reco.runNumber, "%i" % tree_reco.eventNumber, "%f" % weight,
        "%4.1f" % event[0][0], "%4.1f" % event[0][1], "%4.1f" % event[0][2], "%4.1f" % event[0][3], "%4.1f" % event[0][4],  "%.3f" % event[0][5], 
        "%4.1f" % event[1][0], "%4.1f" % event[1][1], "%4.1f" % event[1][2], "%4.1f" % event[1][3], "%4.1f" % event[1][4],  "%.3f" % event[1][5], 
        "%4.1f" % event[2][0], "%4.1f" % event[2][1], "%4.1f" % event[2][2], "%4.1f" % event[2][3], "%4.1f" % event[2][4],  "%.3f" % event[2][5],
        "%4.1f" % event[3][0], "%4.1f" % event[3][1], "%4.1f" % event[3][2], "%4.1f" % event[3][3], "%4.1f" % event[3][4],  "%.3f" % event[3][5], 
        "%4.1f" % event[4][0], "%4.1f" % event[4][1], "%4.1f" % event[4][2], "%4.1f" % event[4][3], "%4.1f" % event[4][4],  "%.3f" % event[4][5], 
        "%4.1f" % event[5][0], "%4.1f" % event[5][1], "%4.1f" % event[5][2], "%4.1f" % event[5][3], "%4.1f" % event[5][4],  "%.3f" % event[5][5],  
        "%4.1f" % event[6][0], "%4.1f" % event[6][1], "%4.1f" % event[6][2], "%4.1f" % event[6][3], "%4.1f" % event[6][4],  "%.3f" % event[6][5], 
        "%4.1f" % (t.Px()/GeV),    "%4.1f" % (t.Py()/GeV),    "%4.1f" % (t.Pz()/GeV),    "%4.1f" % (t.E()/GeV),    "%4.1f" % (t.M()/GeV),
        "%4.1f" % (tbar.Px()/GeV), "%4.1f" % (tbar.Py()/GeV), "%4.1f" % (tbar.Pz()/GeV), "%4.1f" % (tbar.E()/GeV), "%4.1f" % (tbar.M()/GeV),
     ) )

        
