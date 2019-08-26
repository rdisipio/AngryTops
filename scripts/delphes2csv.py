#!/usr/bin/env python
import os
import sys
import csv

import ROOT
from ROOT import TLorentzVector, gROOT, TChain, TVector2
from tree_traversal import GetIndices
from helper_functions import *
import numpy as np
import traceback
from data_augmentation import *

gROOT.SetBatch(True)

from AngryTops.features import *

###############################
# CONSTANTS
GeV = 1e3
TeV = 1e6

# Artificially increase training data size by 5 by rotating events differently 5 different ways
n_data_aug = 1
if len(sys.argv) > 3:
    n_data_aug = int(sys.argv[3])

# Maximum number of entries
n_evt_max = -1
# if len(sys.argv) > 2: n_evt_max = int( sys.argv[2] )

###############################
# BUILDING OUTPUTFILE

# List of filenames
filelistname = sys.argv[1]

# Output filename
outfilename = sys.argv[2]
outfile = open(outfilename, "wt")
csvwriter = csv.writer(outfile)
print ("INFO: output file:", outfilename)

###############################
# BUILDING OUTPUTFILE

tree = TChain("Delphes", "Delphes")
f = open(filelistname, 'r')
for fname in f.readlines():
    fname = fname.strip()
    tree.AddFile(fname)

n_entries = tree.GetEntries()
print("INFO: entries found:", n_entries)

###############################
# LOOPING THROUGH EVENTS

if n_evt_max > 0:
    n_entries = min([n_evt_max, n_entries])

n_jets_per_event = 5
print("INFO: looping over %i reco-level events" % n_entries)
print("INFO: using data augmentation: rotateZ %ix" % n_data_aug)

# Number of events which are actually copied over
n_good = 0

# Looping through the reconstructed entries
for ientry in range(n_entries):

    tree.GetEntry(ientry)
    runNumber = tree.GetTreeNumber()
    eventNumber = tree.GetLeaf("Event.Number").GetValue()
    weight = tree.GetLeaf("Event.Weight").GetValue()

    # Printing how far along in the loop we are
    if (n_entries < 10) or ((ientry+1) % int(float(n_entries)/10.) == 0):
        perc = 100. * ientry / float(n_entries)
        print("INFO: Event %-9i  (%3.0f %%)" % (ientry, perc))

    # Number of muons, leptons, jets and bjets (bjet_n set later)
    mu_n = tree.GetLeaf("Muon.PT").GetLen()
    jets_n = tree.GetLeaf("Jet.PT").GetLen()
    bjets_n = 0

    # If more than one lepton of less than 4 jets, cut
    if mu_n != 1:
        continue
    if jets_n < 4:
        continue

    ##############################################################
    # Muon vector.
    lep = TLorentzVector()
    lep.SetPtEtaPhiM(tree.GetLeaf("Muon.PT").GetValue(0),
                     tree.GetLeaf("Muon.Eta").GetValue(0),
                     tree.GetLeaf("Muon.Phi").GetValue(0),
                     0
                     )
    lep.sumPT = tree.GetLeaf("Muon.SumPt").GetValue(0)
    if lep.Pt() < 20:
        continue  # Fail to get a muon passing the threshold
    if np.abs(lep.Eta()) > 2.5:
        continue

    # Missing Energy values
    met_met = tree.GetLeaf("MissingET.MET").GetValue(0)
    met_phi = tree.GetLeaf("MissingET.Phi").GetValue(0)
    met_eta = tree.GetLeaf("MissingET.Eta").GetValue(0)

    # Append jets, check prob of being a bjet, and update bjet number

    jets = []
    bjets = []
    for i in range(jets_n):
        if i >= n_jets_per_event:
            break

        if tree.GetLeaf("Jet.PT").GetValue(i) < 20.:
            continue
        if np.abs(tree.GetLeaf("Jet.Eta").GetValue(i)) > 2.5:
            continue

        jets += [TLorentzVector()]
        j = jets[-1]
        j.index = i
        j.SetPtEtaPhiM(
            tree.GetLeaf("Jet.PT").GetValue(i),
            tree.GetLeaf("Jet.Eta").GetValue(i),
            tree.GetLeaf("Jet.Phi").GetValue(i),
            tree.GetLeaf("Jet.Mass").GetValue(i))
        j.btag = tree.GetLeaf("Jet.BTag").GetValue(i)
        if j.btag > 0.0:
            bjets_n += 1
            bjets.append(j)

    # Cut based on number of passed jets
    jets_n = len(jets)
    if jets_n < 4:
        continue

    ##############################################################
    # Build output data we are trying to predict with RNN
    try:
        indices = GetIndices(tree, ientry)
    except Exception as e:
        print("Exception thrown when retrieving indices")
        print(e)
        print(traceback.format_exc())
        continue

    t_had = TLorentzVector()
    t_lep = TLorentzVector()
    W_had = TLorentzVector()
    W_lep = TLorentzVector()
    b_had = TLorentzVector()
    b_lep = TLorentzVector()

    t_had.SetPtEtaPhiM(tree.GetLeaf("Particle.PT").GetValue(indices['t_had']),
                       tree.GetLeaf("Particle.Eta").GetValue(indices['t_had']),
                       tree.GetLeaf("Particle.Phi").GetValue(indices['t_had']),
                       tree.GetLeaf("Particle.Mass").GetValue(indices['t_had'])
                       )

    W_had.SetPtEtaPhiM(tree.GetLeaf("Particle.PT").GetValue(indices['W_had']),
                       tree.GetLeaf("Particle.Eta").GetValue(indices['W_had']),
                       tree.GetLeaf("Particle.Phi").GetValue(indices['W_had']),
                       tree.GetLeaf("Particle.Mass").GetValue(indices['W_had'])
                       )

    t_lep.SetPtEtaPhiM(tree.GetLeaf("Particle.PT").GetValue(indices['t_lep']),
                       tree.GetLeaf("Particle.Eta").GetValue(indices['t_lep']),
                       tree.GetLeaf("Particle.Phi").GetValue(indices['t_lep']),
                       tree.GetLeaf("Particle.Mass").GetValue(indices['t_lep']))

    W_lep.SetPtEtaPhiM(tree.GetLeaf("Particle.PT").GetValue(indices['W_lep']),
                       tree.GetLeaf("Particle.Eta").GetValue(indices['W_lep']),
                       tree.GetLeaf("Particle.Phi").GetValue(indices['W_lep']),
                       tree.GetLeaf("Particle.Mass").GetValue(indices['W_lep']))

    b_had.SetPtEtaPhiM(tree.GetLeaf("Particle.PT").GetValue(indices['b_had']),
                       tree.GetLeaf("Particle.Eta").GetValue(indices['b_had']),
                       tree.GetLeaf("Particle.Phi").GetValue(indices['b_had']),
                       tree.GetLeaf("Particle.Mass").GetValue(indices['b_had']))

    b_lep.SetPtEtaPhiM(tree.GetLeaf("Particle.PT").GetValue(indices['b_lep']),
                       tree.GetLeaf("Particle.Eta").GetValue(indices['b_lep']),
                       tree.GetLeaf("Particle.Phi").GetValue(indices['b_lep']),
                       tree.GetLeaf("Particle.Mass").GetValue(indices['b_lep']))

    ##############################################################
    # CUTS USING PARTICLE LEVEL OBJECTS
    if (t_had.Pz() == 0.) or (t_had.M() != t_had.M()):
        print("Invalid t_had values, P_z = {0}, M = {1}".format(
            t_had.Pz(), t_had.M()))
        continue
    if (t_lep.Pz() == 0.) or (t_lep.M() != t_lep.M()):
        print("Invalid t_lep values, P_z = {0}, M = {1}".format(
            t_lep.Pz(), t_lep.M()))
        continue

    ##############################################################
    # DERIVED EVENT-WISE QUANTITES
    # Sum of all Pt's
    H_t = tree.GetLeaf("ScalarHT.HT").GetValue(0)

    vertex_n = tree.GetLeaf("Vertex_size").GetValue(0)

    ##############################################################
    # PRINT OUT TO FILE
    csvwriter.writerow((
        "%i" % runNumber, "%i" % eventNumber, "%.5f" % weight, "%i" % jets_n, "%i" % bjets_n,
        "%f" % lep.Px(), "%f" % lep.Py(), "%f" % lep.Pz(), "%f" % met_met, "%f" % met_phi,
        "%f" % jets[0][0], "%f" % jets[0][1], "%f" % jets[0][2], "%f" % jets[0][3], "%f" % jets[0][4], "%f" % jets[0][5],
        "%f" % jets[1][0], "%f" % jets[1][1], "%f" % jets[1][2], "%f" % jets[1][3], "%f" % jets[1][4], "%f" % jets[1][5],
        "%f" % jets[2][0], "%f" % jets[2][1], "%f" % jets[2][2], "%f" % jets[2][3], "%f" % jets[2][4], "%f" % jets[2][5],
        "%f" % jets[3][0], "%f" % jets[3][1], "%f" % jets[3][2], "%f" % jets[3][3], "%f" % jets[3][4], "%f" % jets[3][5],
        "%f" % jets[4][0], "%f" % jets[4][1], "%f" % jets[4][2], "%f" % jets[4][3], "%f" % jets[4][4], "%f" % jets[4][5],
        "%f" % t_lep.Px(),  "%f" % t_lep.Py(), "%f" % t_lep.Pz(
        ), "%f" % t_lep.E(), "%f" % t_lep.M(),
        "%f" % t_had.Px(),  "%f" % t_had.Py(), "%f" % t_had.Pz(
        ), "%f" % t_had.E(), "%f" % t_had.M(),
        "%f" % W_lep.Px(),  "%f" % W_lep.Py(), "%f" % W_lep.Pz(
        ), "%f" % W_lep.E(), "%f" % W_lep.M(),
        "%f" % W_had.Px(),  "%f" % W_had.Py(), "%f" % W_had.Pz(
        ), "%f" % W_had.E(), "%f" % W_had.M(),
        "%f" % b_lep.Px(),  "%f" % b_lep.Py(), "%f" % b_lep.Pz(
        ), "%f" % b_lep.E(), "%f" % b_lep.M(),
        "%f" % b_had.Px(),  "%f" % b_had.Py(), "%f" % b_had.Pz(
        ), "%f" % b_had.E(), "%f" % b_had.M(),
        )


##############################################################
# Close Program
outfile.close()

f_good=100. * n_good / n_entries
print("INFO: output file:", outfilename)
print("INFO: %i entries written (%.2f %%)" % (n_good, f_good))
