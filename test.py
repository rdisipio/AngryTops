#!/usr/bin/env python

GeV = 1e3
TeV = 1e6
m_t = 172.5
m_W = 80.4
m_b = 4.95

import os, sys, time
import argparse

from keras.models import load_model

from ROOT import *
from array import array
import cPickle as pickle
import numpy as np
import pandas as pd

np.set_printoptions( precision=3, suppress=True, linewidth=250 )

from features import *
import models

################

def MakeP4( y, m=0. ):
  p4 = TLorentzVector()

  px = y[0]
  py = y[1]
  pz = y[2]
  P2 = px*px + py*py + pz*pz
  E  = TMath.Sqrt( P2 + m*m )
  
  p4.SetPxPyPzE( px, py, pz, E )
  return p4


###############

infilename = "csv/topreco_mc.410501.nominal.csv"
if len(sys.argv) > 1: infilename = sys.argv[1]

##################
# Load Keras stuff
X_scaler = None
y_scaler = None
dnn      = None

model_filename  = "keras/model.rnn.PxPyPzEMBw.h5"
scaler_filename = "keras/scaler.rnn.PxPyPzEMBw.pkl"

dnn = load_model( model_filename )
print dnn.summary()

with open( scaler_filename, "rb" ) as file_scaler:
  X_t_lep_scaler = pickle.load( file_scaler )
  X_t_had_scaler = pickle.load( file_scaler )
  y_scaler = pickle.load( file_scaler )

# read in input file
data = pd.read_csv( infilename, delimiter=',', names=header )

X_jets   = data[input_features_jets].values
X_lepton = data[input_features_lep].values

y_true_W_lep = data[target_features_W_lep].values
y_true_W_had = data[target_features_W_had].values
y_true_b_lep = data[target_features_b_lep].values
y_true_b_had = data[target_features_b_had].values
y_true_t_lep = data[target_features_t_lep].values
y_true_t_had = data[target_features_t_had].values

event_info = data[features_event_info].values
n_events   = len(event_info)

X_jets   = X_jets.reshape( (n_events,n_jets_per_event,n_features_per_jet) )

print "INFO: fitting ttbar decay chain..."
y_fitted = dnn.predict( { 'jets_input':X_jets, 'lepton_input':X_lepton } )
#y_fitted = y_scaler.inverse_transform( y_fitted )
print "INFO ...done"

#print "INFO: shape of fitted particles:", y_fitted.shape
#y_fitted = y_fitted.reshape( (4, 3) )
#print y_fitted

# open output file
ofilename = "output/testing.root"
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

#################
# book histograms
histograms = {}

# basic distributions

# True
histograms['W_had_pt_true']       = TH1F( "W_had_pt_true",  ";Hadronic W p_{T} [GeV]", 50, 0., 1500. )
histograms['W_had_y_true']        = TH1F( "W_had_y_true",   ";Hadronic W #eta", 25, -5., 5. )
histograms['W_had_phi_true']      = TH1F( "W_had_phi_true", ";Hadronic W #phi", 32, -3.2, 3.2 )
histograms['W_had_E_true']        = TH1F( "W_had_E_true",   ";Hadronic W E [GeV]", 50, 0., 1500. )
histograms['W_had_m_true']        = TH1F( "W_had_m_true",   ";Hadronic W m [GeV]", 30, 0., 300.  )

histograms['b_had_pt_true']       = TH1F( "b_had_pt_true",  ";Hadronic b p_{T} [GeV]", 50, 0., 1500. )
histograms['b_had_y_true']        = TH1F( "b_had_y_true",   ";Hadronic b #eta", 25, -5., 5. )
histograms['b_had_phi_true']      = TH1F( "b_had_phi_true", ";Hadronic b #phi", 32, -3.2, 3.2 )
histograms['b_had_E_true']        = TH1F( "b_had_E_true",   ";Hadronic b E [GeV]", 50, 0., 1500. )
histograms['b_had_m_true']        = TH1F( "b_had_m_true",   ";Hadronic b m [GeV]", 30, 0., 300.  )

histograms['t_had_pt_true']       = TH1F( "t_had_pt_true",  ";Hadronic top p_{T} [GeV]", 50, 0., 1500. )
histograms['t_had_y_true']        = TH1F( "t_had_y_true",   ";Hadronic top #eta", 25, -5., 5. )
histograms['t_had_phi_true']      = TH1F( "t_had_phi_true", ";Hadronic top #phi", 32, -3.2, 3.2 )
histograms['t_had_E_true']        = TH1F( "t_had_E_true",   ";Hadronic top E [GeV]", 50, 0., 1500. )
histograms['t_had_m_true']        = TH1F( "t_had_m_true",   ";Hadronic top m [GeV]", 30, 0., 300.  )

histograms['W_lep_pt_true']       = TH1F( "W_lep_pt_true",   ";Leptonic W p_{T} [GeV]", 50, 0., 1500. )
histograms['W_lep_y_true']        = TH1F( "W_lep_y_true",    ";Leptonic W #eta", 25, -5., 5. )
histograms['W_lep_phi_true']      = TH1F( "W_lep_phi_true",  ";Leptonic W #phi", 32, -3.2, 3.2 )
histograms['W_lep_E_true']        = TH1F( "W_lep_E_true",    ";Leptonic W E [GeV]", 50, 0., 1500. )
histograms['W_lep_m_true']        = TH1F( "W_lep_m_true",    ";Leptonic W m [GeV]", 30, 0., 300. )

histograms['b_lep_pt_true']       = TH1F( "b_lep_pt_true",   ";Leptonic b p_{T} [GeV]", 50, 0., 1500. )
histograms['b_lep_y_true']        = TH1F( "b_lep_y_true",    ";Leptonic b #eta", 25, -5., 5. )
histograms['b_lep_phi_true']      = TH1F( "b_lep_phi_true",  ";Leptonic b #phi", 32, -3.2, 3.2 )
histograms['b_lep_E_true']        = TH1F( "b_lep_E_true",    ";Leptonic b E [GeV]", 50, 0., 1500. )
histograms['b_lep_m_true']        = TH1F( "b_lep_m_true",    ";Leptonic b m [GeV]", 30, 0., 300. )

histograms['t_lep_pt_true']       = TH1F( "t_lep_pt_true",   ";Leptonic top p_{T} [GeV]", 50, 0., 1500. )
histograms['t_lep_y_true']        = TH1F( "t_lep_y_true",    ";Leptonic top #eta", 25, -5., 5. )
histograms['t_lep_phi_true']      = TH1F( "t_lep_phi_true",  ";Leptonic top #phi", 32, -3.2, 3.2 )
histograms['t_lep_E_true']        = TH1F( "t_lep_E_true",    ";Leptonic top E [GeV]", 50, 0., 1500. )
histograms['t_lep_m_true']        = TH1F( "t_lep_m_true",    ";Leptonic top m [GeV]", 30, 0., 300. )

# Fitted
histograms['W_had_pt_fitted']       = TH1F( "W_had_pt_fitted",  ";Hadronic W p_{T} [GeV]", 50, 0., 1500. )
histograms['W_had_y_fitted']        = TH1F( "W_had_y_fitted",   ";Hadronic W #eta", 25, -5., 5. )
histograms['W_had_phi_fitted']      = TH1F( "W_had_phi_fitted", ";Hadronic W #phi", 32, -3.2, 3.2 )
histograms['W_had_E_fitted']        = TH1F( "W_had_E_fitted",   ";Hadronic W E [GeV]", 50, 0., 1500. )
histograms['W_had_m_fitted']        = TH1F( "W_had_m_fitted",   ";Hadronic W m [GeV]", 30, 0., 300.  )

histograms['b_had_pt_fitted']       = TH1F( "b_had_pt_fitted",  ";Hadronic b p_{T} [GeV]", 50, 0., 1500. )
histograms['b_had_y_fitted']        = TH1F( "b_had_y_fitted",   ";Hadronic b #eta", 25, -5., 5. )
histograms['b_had_phi_fitted']      = TH1F( "b_had_phi_fitted", ";Hadronic b #phi", 32, -3.2, 3.2 )
histograms['b_had_E_fitted']        = TH1F( "b_had_E_fitted",   ";Hadronic b E [GeV]", 50, 0., 1500. )
histograms['b_had_m_fitted']        = TH1F( "b_had_m_fitted",   ";Hadronic b m [GeV]", 30, 0., 300.  )

histograms['t_had_pt_fitted']       = TH1F( "t_had_pt_fitted",  ";Hadronic top p_{T} [GeV]", 50, 0., 1500. )
histograms['t_had_y_fitted']        = TH1F( "t_had_y_fitted",   ";Hadronic top #eta", 25, -5., 5. )
histograms['t_had_phi_fitted']      = TH1F( "t_had_phi_fitted", ";Hadronic top #phi", 32, -3.2, 3.2 )
histograms['t_had_E_fitted']        = TH1F( "t_had_E_fitted",   ";Hadronic top E [GeV]", 50, 0., 1500. )
histograms['t_had_m_fitted']        = TH1F( "t_had_m_fitted",   ";Hadronic top m [GeV]", 30, 0., 300.  )

histograms['W_lep_pt_fitted']       = TH1F( "W_lep_pt_fitted",   ";Leptonic W p_{T} [GeV]", 50, 0., 1500. )
histograms['W_lep_y_fitted']        = TH1F( "W_lep_y_fitted",    ";Leptonic W #eta", 25, -5., 5. )
histograms['W_lep_phi_fitted']      = TH1F( "W_lep_phi_fitted",  ";Leptonic W #phi", 32, -3.2, 3.2 )
histograms['W_lep_E_fitted']        = TH1F( "W_lep_E_fitted",    ";Leptonic W E [GeV]", 50, 0., 1500. )
histograms['W_lep_m_fitted']        = TH1F( "W_lep_m_fitted",    ";Leptonic W m [GeV]", 30, 0., 300. )

histograms['b_lep_pt_fitted']       = TH1F( "b_lep_pt_fitted",   ";Leptonic b p_{T} [GeV]", 50, 0., 1500. )
histograms['b_lep_y_fitted']        = TH1F( "b_lep_y_fitted",    ";Leptonic b #eta", 25, -5., 5. )
histograms['b_lep_phi_fitted']      = TH1F( "b_lep_phi_fitted",  ";Leptonic b #phi", 32, -3.2, 3.2 )
histograms['b_lep_E_fitted']        = TH1F( "b_lep_E_fitted",    ";Leptonic b E [GeV]", 50, 0., 1500. )
histograms['b_lep_m_fitted']        = TH1F( "b_lep_m_fitted",    ";Leptonic b m [GeV]", 30, 0., 300. )

histograms['t_lep_pt_fitted']       = TH1F( "t_lep_pt_fitted",   ";Leptonic top p_{T} [GeV]", 50, 0., 1500. )
histograms['t_lep_y_fitted']        = TH1F( "t_lep_y_fitted",    ";Leptonic top #eta", 25, -5., 5. )
histograms['t_lep_phi_fitted']      = TH1F( "t_lep_phi_fitted",  ";Leptonic top #phi", 32, -3.2, 3.2 )
histograms['t_lep_E_fitted']        = TH1F( "t_lep_E_fitted",    ";Leptonic top E [GeV]", 50, 0., 1500. )
histograms['t_lep_m_fitted']        = TH1F( "t_lep_m_fitted",    ";Leptonic top m [GeV]", 30, 0., 300. )

# 2D correlations 
histograms['corr_t_had_pt']    = TH2F( "corr_t_had_pt",      ";True Hadronic top p_{T} [GeV];Fitted Hadronic top p_{T} [GeV]", 50, 0., 1500., 50, 0., 1500. )
histograms['corr_t_had_y']     = TH2F( "corr_t_had_y",       ";True Hadronic top y;Fitted Hadronic top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_t_had_phi']   = TH2F( "corr_t_had_phi",     ";True Hadronic top #phi;Fitted Hadronic top #phi", 32, -3.2, 3.2, 32, -3.2, 3.2 )
histograms['corr_t_had_E']     = TH2F( "corr_t_had_E",       ";True Hadronic top E [GeV];Fitted Hadronic top E [GeV]", 50, 0., 1500., 50, 0., 1500. )
histograms['corr_t_had_m']     = TH2F( "corr_t_had_m",       ";True Hadronic top m [GeV];Fitted Hadronic top m [GeV]", 25, 170., 175., 20, 150., 250. )

histograms['corr_t_lep_pt']    = TH2F( "corr_t_lep_pt",     ";True Leptonic top p_{T} [GeV];Fitted Leptonic top p_{T} [GeV]", 50, 0., 1500., 50, 0., 1500. )
histograms['corr_t_lep_y']     = TH2F( "corr_t_lep_y",      ";True Leptonic top y;Fitted Leptonic top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_t_lep_phi']   = TH2F( "corr_t_lep_phi",    ";True Leptonic top #phi;Fitted Leptonic top #phi", 32, -3.2, 3.2, 32, -3.2, 3.2 )
histograms['corr_t_lep_E']     = TH2F( "corr_t_lep_E",      ";True Leptonic top E [GeV];Fitted Leptonic top E [GeV]", 50, 0., 1500., 50, 0., 1500. )
histograms['corr_t_lep_m']     = TH2F( "corr_t_lep_m",      ";True Leptonic top m [GeV];Fitted Leptonic top m [GeV]", 25, 170., 175., 20, 150., 250. )

# resolution
histograms['reso_W_had_px']   = TH1F( "reso_W_had_px",   ";Hadronic W p_{x} resolution", 100, -3.0, 3.0 )
histograms['reso_W_had_py']   = TH1F( "reso_W_had_py",   ";Hadronic W p_{y} resolution", 100, -3.0, 3.0 )
histograms['reso_W_had_pz']   = TH1F( "reso_W_had_pz",   ";Hadronic W p_{z} resolution", 100, -3.0, 3.0 )
histograms['reso_W_had_pt']   = TH1F( "reso_W_had_pt",   ";Hadronic W p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_W_had_y']    = TH1F( "reso_W_had_y",    ";Hadronic W y resolution",     100, -3.0, 3.0 )
histograms['reso_W_had_phi']  = TH1F( "reso_W_had_phi",  ";Hadronic W #phi resolution",  100, -3.0, 3.0 )
histograms['reso_W_had_E']    = TH1F( "reso_W_had_E",    ";Hadronic W E resolution",     100, -3.0, 3.0 )
histograms['reso_W_had_m']    = TH1F( "reso_W_had_m",    ";Hadronic W M resolution",     100, -3.0, 3.0 )

histograms['reso_b_had_px']   = TH1F( "reso_b_had_px",   ";Hadronic b p_{x} resolution", 100, -3.0, 3.0 )
histograms['reso_b_had_py']   = TH1F( "reso_b_had_py",   ";Hadronic b p_{y} resolution", 100, -3.0, 3.0 )
histograms['reso_b_had_pz']   = TH1F( "reso_b_had_pz",   ";Hadronic b p_{z} resolution", 100, -3.0, 3.0 )
histograms['reso_b_had_pt']   = TH1F( "reso_b_had_pt",   ";Hadronic b p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_b_had_y']    = TH1F( "reso_b_had_y",    ";Hadronic b y resolution",     100, -3.0, 3.0 )
histograms['reso_b_had_phi']  = TH1F( "reso_b_had_phi",  ";Hadronic b #phi resolution",  100, -3.0, 3.0 )
histograms['reso_b_had_E']    = TH1F( "reso_b_had_E",    ";Hadronic b E resolution",     100, -3.0, 3.0 )
histograms['reso_b_had_m']    = TH1F( "reso_b_had_m",    ";Hadronic b M resolution",     100, -3.0, 3.0 )

histograms['reso_t_had_px']   = TH1F( "reso_t_had_px",   ";Hadronic top p_{x} resolution", 100, -3.0, 3.0 )
histograms['reso_t_had_py']   = TH1F( "reso_t_had_py",   ";Hadronic top p_{y} resolution", 100, -3.0, 3.0 )
histograms['reso_t_had_pz']   = TH1F( "reso_t_had_pz",   ";Hadronic top p_{z} resolution", 100, -3.0, 3.0 )
histograms['reso_t_had_pt']   = TH1F( "reso_t_had_pt",   ";Hadronic top p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_t_had_y']    = TH1F( "reso_t_had_y",    ";Hadronic top y resolution",  100, -3.0, 3.0 )
histograms['reso_t_had_phi']  = TH1F( "reso_t_had_phi",  ";Hadronic top #phi resolution",  100, -3.0, 3.0 )
histograms['reso_t_had_E']    = TH1F( "reso_t_had_E",    ";Hadronic top E resolution",     100, -3.0, 3.0 )
histograms['reso_t_had_m']    = TH1F( "reso_t_had_m",    ";Hadronic top M resolution",     100, -3.0, 3.0 )

histograms['reso_W_lep_px']  = TH1F( "reso_W_lep_px",  ";Leptonic W p_{x} resolution", 100, -3.0, 3.0 )
histograms['reso_W_lep_py']  = TH1F( "reso_W_lep_py",  ";Leptonic W p_{y} resolution", 100, -3.0, 3.0 )
histograms['reso_W_lep_pz']  = TH1F( "reso_W_lep_pz",  ";Leptonic W p_{z} resolution", 100, -3.0, 3.0 )
histograms['reso_W_lep_pt']  = TH1F( "reso_W_lep_pt",  ";Leptonic W p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_W_lep_y']   = TH1F( "reso_W_lep_y",   ";Leptonic W y resolution",  100, -3.0, 3.0 )
histograms['reso_W_lep_phi'] = TH1F( "reso_W_lep_phi", ";Leptonic W #phi resolution",  100, -3.0, 3.0 )
histograms['reso_W_lep_E']   = TH1F( "reso_W_lep_E",   ";Leptonic W E resolution",     100, -3.0, 3.0 )
histograms['reso_W_lep_m']   = TH1F( "reso_W_lep_m",   ";Leptonic W M resolution",     100, -3.0, 3.0 )

histograms['reso_b_lep_px']  = TH1F( "reso_b_lep_px",  ";Leptonic b p_{x} resolution", 100, -3.0, 3.0 )
histograms['reso_b_lep_py']  = TH1F( "reso_b_lep_py",  ";Leptonic b p_{y} resolution", 100, -3.0, 3.0 )
histograms['reso_b_lep_pz']  = TH1F( "reso_b_lep_pz",  ";Leptonic b p_{z} resolution", 100, -3.0, 3.0 )
histograms['reso_b_lep_pt']  = TH1F( "reso_b_lep_pt",  ";Leptonic b p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_b_lep_y']   = TH1F( "reso_b_lep_y",   ";Leptonic b y resolution",  100, -3.0, 3.0 )
histograms['reso_b_lep_phi'] = TH1F( "reso_b_lep_phi", ";Leptonic b #phi resolution",  100, -3.0, 3.0 )
histograms['reso_b_lep_E']   = TH1F( "reso_b_lep_E",   ";Leptonic b E resolution",     100, -3.0, 3.0 )
histograms['reso_b_lep_m']   = TH1F( "reso_b_lep_m",   ";Leptonic b M resolution",     100, -3.0, 3.0 )

histograms['reso_t_lep_px']  = TH1F( "reso_t_lep_px",  ";Leptonic top p_{x} resolution", 100, -3.0, 3.0 )
histograms['reso_t_lep_py']  = TH1F( "reso_t_lep_py",  ";Leptonic top p_{y} resolution", 100, -3.0, 3.0 )
histograms['reso_t_lep_pz']  = TH1F( "reso_t_lep_pz",  ";Leptonic top p_{z} resolution", 100, -3.0, 3.0 )
histograms['reso_t_lep_pt']  = TH1F( "reso_t_lep_pt",  ";Leptonic top p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_t_lep_y']   = TH1F( "reso_t_lep_y",   ";Leptonic top y resolution",  100, -3.0, 3.0 )
histograms['reso_t_lep_phi'] = TH1F( "reso_t_lep_phi", ";Leptonic top #phi resolution",  100, -3.0, 3.0 )
histograms['reso_t_lep_E']   = TH1F( "reso_t_lep_E",   ";Leptonic top E resolution",     100, -3.0, 3.0 )
histograms['reso_t_lep_m']   = TH1F( "reso_t_lep_m",   ";Leptonic top M resolution",     100, -3.0, 3.0 )

for h in histograms.values(): h.Sumw2()


print "INFO: starting event loop. Found %i events" % n_events
n_good = 0
# Print out example
for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print "INFO: Event %-9i  (%3.0f %%)" % ( i, perc )

    w = event_info[i][2]
    jets_n  = event_info[i][3]
    bjets_n = event_info[i][4]

    W_lep_true   = MakeP4( y_true_W_lep[i], m_W )
    W_lep_fitted = MakeP4( y_fitted[0][i],  m_W )

    W_had_true   = MakeP4( y_true_W_had[i], m_W )
    W_had_fitted = MakeP4( y_fitted[1][i],  m_W )

    b_lep_true   = MakeP4( y_true_b_lep[i], m_b )
    b_lep_fitted = MakeP4( y_fitted[2][i],  m_b )

    b_had_true   = MakeP4( y_true_b_had[i], m_b )
    b_had_fitted = MakeP4( y_fitted[3][i],  m_b )

    t_lep_true   = MakeP4( y_true_t_lep[i], m_t )
    t_lep_fitted = MakeP4( y_fitted[4][i],  m_t )

    t_had_true   = MakeP4( y_true_t_had[i], m_t )
    t_had_fitted = MakeP4( y_fitted[5][i],  m_t )

    try:
        reso_W_had_px  = ( W_had_fitted.Px()  - W_had_true.Px()  ) / W_had_true.Px()
        reso_W_had_py  = ( W_had_fitted.Py()  - W_had_true.Py()  ) / W_had_true.Py()
        reso_W_had_pz  = ( W_had_fitted.Pz()  - W_had_true.Pz()  ) / W_had_true.Pz()
        reso_W_had_pt  = ( W_had_fitted.Pt()  - W_had_true.Pt()  ) / W_had_true.Pt()   
        reso_W_had_y   = ( W_had_fitted.Rapidity() - W_had_true.Rapidity() ) / W_had_true.Rapidity()  
        reso_W_had_phi = ( W_had_fitted.Phi() - W_had_true.Phi() ) / W_had_true.Phi() 
        reso_W_had_E   = ( W_had_fitted.E()   - W_had_true.E()   ) / W_had_true.E()   
        reso_W_had_m   = ( W_had_fitted.M()   - W_had_true.M()   ) / W_had_true.M()

        reso_b_had_px  = ( b_had_fitted.Px()  - b_had_true.Px()  ) / b_had_true.Px()
        reso_b_had_py  = ( b_had_fitted.Py()  - b_had_true.Py()  ) / b_had_true.Py()
        reso_b_had_pz  = ( b_had_fitted.Pz()  - b_had_true.Pz()  ) / b_had_true.Pz()
        reso_b_had_pt  = ( b_had_fitted.Pt()  - b_had_true.Pt()  ) / b_had_true.Pt()   
        reso_b_had_y   = ( b_had_fitted.Rapidity() - b_had_true.Rapidity() ) / b_had_true.Rapidity()  
        reso_b_had_phi = ( b_had_fitted.Phi() - b_had_true.Phi() ) / b_had_true.Phi() 
        reso_b_had_E   = ( b_had_fitted.E()   - b_had_true.E()   ) / b_had_true.E()   
        reso_b_had_m   = ( b_had_fitted.M()   - b_had_true.M()   ) / b_had_true.M()

        reso_t_had_px  = ( t_had_fitted.Px()  - t_had_true.Px()  ) / t_had_true.Px()
        reso_t_had_py  = ( t_had_fitted.Py()  - t_had_true.Py()  ) / t_had_true.Py()
        reso_t_had_pz  = ( t_had_fitted.Pz()  - t_had_true.Pz()  ) / t_had_true.Pz()
        reso_t_had_pt  = ( t_had_fitted.Pt()  - t_had_true.Pt()  ) / t_had_true.Pt()   
        reso_t_had_y   = ( t_had_fitted.Rapidity() - t_had_true.Rapidity() ) / t_had_true.Rapidity()  
        reso_t_had_phi = ( t_had_fitted.Phi() - t_had_true.Phi() ) / t_had_true.Phi() 
        reso_t_had_E   = ( t_had_fitted.E()   - t_had_true.E()   ) / t_had_true.E()   
        reso_t_had_m   = ( t_had_fitted.M()   - t_had_true.M()   ) / t_had_true.M()  
    except:
        print "WARNING: invalid hadronic top, skipping event ( rn=%-10i en=%-10i )" % ( event_info[i][0], event_info[i][1] )
        continue

    try:
        reso_W_lep_px  = ( W_lep_fitted.Px()  - W_lep_true.Px()  ) / W_lep_true.Px()
        reso_W_lep_py  = ( W_lep_fitted.Py()  - W_lep_true.Py()  ) / W_lep_true.Py()
        reso_W_lep_pz  = ( W_lep_fitted.Pz()  - W_lep_true.Pz()  ) / W_lep_true.Pz()
        reso_W_lep_pt  = ( W_lep_fitted.Pt()  - W_lep_true.Pt()  ) / W_lep_true.Pt()   
        reso_W_lep_y   = ( W_lep_fitted.Rapidity() - W_lep_true.Rapidity() ) / W_lep_true.Rapidity()  
        reso_W_lep_phi = ( W_lep_fitted.Phi() - W_lep_true.Phi() ) / W_lep_true.Phi()   
        reso_W_lep_E   = ( W_lep_fitted.E()   - W_lep_true.E()   ) / W_lep_true.E() 
        reso_W_lep_m   = ( W_lep_fitted.M()   - W_lep_true.M()   ) / W_lep_true.M()

        reso_b_lep_px  = ( b_lep_fitted.Px()  - b_lep_true.Px()  ) / b_lep_true.Px()
        reso_b_lep_py  = ( b_lep_fitted.Py()  - b_lep_true.Py()  ) / b_lep_true.Py()
        reso_b_lep_pz  = ( b_lep_fitted.Pz()  - b_lep_true.Pz()  ) / b_lep_true.Pz()
        reso_b_lep_pt  = ( b_lep_fitted.Pt()  - b_lep_true.Pt()  ) / b_lep_true.Pt()   
        reso_b_lep_y   = ( b_lep_fitted.Rapidity() - b_lep_true.Rapidity() ) / b_lep_true.Rapidity()  
        reso_b_lep_phi = ( b_lep_fitted.Phi() - b_lep_true.Phi() ) / b_lep_true.Phi()   
        reso_b_lep_E   = ( b_lep_fitted.E()   - b_lep_true.E()   ) / b_lep_true.E() 
        reso_b_lep_m   = ( b_lep_fitted.M()   - b_lep_true.M()   ) / b_lep_true.M()

        reso_t_lep_px  = ( t_lep_fitted.Px()  - t_lep_true.Px()  ) / t_lep_true.Px()
        reso_t_lep_py  = ( t_lep_fitted.Py()  - t_lep_true.Py()  ) / t_lep_true.Py()
        reso_t_lep_pz  = ( t_lep_fitted.Pz()  - t_lep_true.Pz()  ) / t_lep_true.Pz()
        reso_t_lep_pt  = ( t_lep_fitted.Pt()  - t_lep_true.Pt()  ) / t_lep_true.Pt()   
        reso_t_lep_y = ( t_lep_fitted.Rapidity() - t_lep_true.Rapidity() ) / t_lep_true.Rapidity()  
        reso_t_lep_phi = ( t_lep_fitted.Phi() - t_lep_true.Phi() ) / t_lep_true.Phi()   
        reso_t_lep_E   = ( t_lep_fitted.E()   - t_lep_true.E()   ) / t_lep_true.E() 
        reso_t_lep_m   = ( t_lep_fitted.M()   - t_lep_true.M()   ) / t_lep_true.M()
    except:
        print "WARNING: invalid leptonic top, skipping event ( rn=%-10i en=%-10i )" % ( event_info[i][0], event_info[i][1] )
        continue

    # true
    histograms['W_had_pt_true'].Fill(  W_had_true.Pt(),  w )
    histograms['W_had_y_true'].Fill(   W_had_true.Rapidity(), w )
    histograms['W_had_phi_true'].Fill( W_had_true.Phi(), w )
    histograms['W_had_E_true'].Fill(   W_had_true.E(),   w )
    histograms['W_had_m_true'].Fill(   W_had_true.M(),   w )

    histograms['b_had_pt_true'].Fill(  b_had_true.Pt(),  w )
    histograms['b_had_y_true'].Fill(   b_had_true.Rapidity(), w )
    histograms['b_had_phi_true'].Fill( b_had_true.Phi(), w )
    histograms['b_had_E_true'].Fill(   b_had_true.E(),   w )
    histograms['b_had_m_true'].Fill(   b_had_true.M(),   w )

    histograms['t_had_pt_true'].Fill(  t_had_true.Pt(),  w )
    histograms['t_had_y_true'].Fill(   t_had_true.Rapidity(), w )
    histograms['t_had_phi_true'].Fill( t_had_true.Phi(), w )
    histograms['t_had_E_true'].Fill(   t_had_true.E(),   w )
    histograms['t_had_m_true'].Fill(   t_had_true.M(),   w )

    histograms['W_lep_pt_true'].Fill(  W_lep_true.Pt(),  w )
    histograms['W_lep_y_true'].Fill(   W_lep_true.Rapidity(), w )
    histograms['W_lep_phi_true'].Fill( W_lep_true.Phi(), w )
    histograms['W_lep_E_true'].Fill(   W_lep_true.E(),   w )
    histograms['W_lep_m_true'].Fill(   W_lep_true.M(),   w )

    histograms['b_lep_pt_true'].Fill(  b_lep_true.Pt(),  w )
    histograms['b_lep_y_true'].Fill(   b_lep_true.Rapidity(), w )
    histograms['b_lep_phi_true'].Fill( b_lep_true.Phi(), w )
    histograms['b_lep_E_true'].Fill(   b_lep_true.E(),   w )
    histograms['b_lep_m_true'].Fill(   b_lep_true.M(),   w )

    histograms['t_lep_pt_true'].Fill(  t_lep_true.Pt(),  w )
    histograms['t_lep_y_true'].Fill(   t_lep_true.Rapidity(), w )
    histograms['t_lep_phi_true'].Fill( t_lep_true.Phi(), w )
    histograms['t_lep_E_true'].Fill(   t_lep_true.E(),   w )
    histograms['t_lep_m_true'].Fill(   t_lep_true.M(),   w )

    # Fitted
    histograms['W_had_pt_fitted'].Fill(  W_had_fitted.Pt(),  w )
    histograms['W_had_y_fitted'].Fill(   W_had_fitted.Rapidity(), w )
    histograms['W_had_phi_fitted'].Fill( W_had_fitted.Phi(), w )
    histograms['W_had_E_fitted'].Fill(   W_had_fitted.E(),   w )
    histograms['W_had_m_fitted'].Fill(   W_had_fitted.M(),   w )

    histograms['b_had_pt_fitted'].Fill(  b_had_fitted.Pt(),  w )
    histograms['b_had_y_fitted'].Fill(   b_had_fitted.Rapidity(), w )
    histograms['b_had_phi_fitted'].Fill( b_had_fitted.Phi(), w )
    histograms['b_had_E_fitted'].Fill(   b_had_fitted.E(),   w )
    histograms['b_had_m_fitted'].Fill(   b_had_fitted.M(),   w )

    histograms['t_had_pt_fitted'].Fill(  t_had_fitted.Pt(),  w )
    histograms['t_had_y_fitted'].Fill(   t_had_fitted.Rapidity(), w )
    histograms['t_had_phi_fitted'].Fill( t_had_fitted.Phi(), w )
    histograms['t_had_E_fitted'].Fill(   t_had_fitted.E(),   w )
    histograms['t_had_m_fitted'].Fill(   t_had_fitted.M(),   w )
    
    histograms['W_lep_pt_fitted'].Fill(  W_lep_fitted.Pt(),  w )
    histograms['W_lep_y_fitted'].Fill(   W_lep_fitted.Rapidity(), w )
    histograms['W_lep_phi_fitted'].Fill( W_lep_fitted.Phi(), w )
    histograms['W_lep_E_fitted'].Fill(   W_lep_fitted.E(),   w )
    histograms['W_lep_m_fitted'].Fill(   W_lep_fitted.M(),   w )

    histograms['b_lep_pt_fitted'].Fill(  b_lep_fitted.Pt(),  w )
    histograms['b_lep_y_fitted'].Fill(   b_lep_fitted.Rapidity(), w )
    histograms['b_lep_phi_fitted'].Fill( b_lep_fitted.Phi(), w )
    histograms['b_lep_E_fitted'].Fill(   b_lep_fitted.E(),   w )
    histograms['b_lep_m_fitted'].Fill(   b_lep_fitted.M(),   w )

    histograms['t_lep_pt_fitted'].Fill(  t_lep_fitted.Pt(),  w )
    histograms['t_lep_y_fitted'].Fill(   t_lep_fitted.Rapidity(), w )
    histograms['t_lep_phi_fitted'].Fill( t_lep_fitted.Phi(), w )
    histograms['t_lep_E_fitted'].Fill(   t_lep_fitted.E(),   w )
    histograms['t_lep_m_fitted'].Fill(   t_lep_fitted.M(),   w )

    histograms['corr_t_had_pt'].Fill(   t_had_true.Pt(),       t_had_fitted.Pt(),  w )
    histograms['corr_t_had_y'].Fill(    t_had_true.Rapidity(), t_had_fitted.Rapidity(), w )
    histograms['corr_t_had_phi'].Fill(  t_had_true.Phi(),      t_had_fitted.Phi(), w )
    histograms['corr_t_had_E'].Fill(    t_had_true.E(),        t_had_fitted.E(),   w )
    histograms['corr_t_had_m'].Fill(    t_had_true.M(),        t_had_fitted.M(),   w )
   
    histograms['corr_t_lep_pt'].Fill(  t_lep_true.Pt(),       t_lep_fitted.Pt(),  w )
    histograms['corr_t_lep_y'].Fill(   t_lep_true.Rapidity(), t_lep_fitted.Rapidity(), w )
    histograms['corr_t_lep_phi'].Fill( t_lep_true.Phi(),      t_lep_fitted.Phi(), w )
    histograms['corr_t_lep_E'].Fill(   t_lep_true.E(),        t_lep_fitted.E(),   w )
    histograms['corr_t_lep_m'].Fill(   t_lep_true.M(),        t_lep_fitted.M(),   w )

    histograms['reso_W_had_px'].Fill(  reso_W_had_px,  w )
    histograms['reso_W_had_py'].Fill(  reso_W_had_py,  w )
    histograms['reso_W_had_pz'].Fill(  reso_W_had_pz,  w )
    histograms['reso_W_had_pt'].Fill(  reso_W_had_pt,  w )
    histograms['reso_W_had_y'].Fill(   reso_W_had_y,   w )
    histograms['reso_W_had_phi'].Fill( reso_W_had_phi, w )
    histograms['reso_W_had_E'].Fill(   reso_W_had_E,   w )
    histograms['reso_W_had_m'].Fill(   reso_W_had_m,   w )

    histograms['reso_b_had_px'].Fill(  reso_b_had_px,  w )
    histograms['reso_b_had_py'].Fill(  reso_b_had_py,  w )
    histograms['reso_b_had_pz'].Fill(  reso_b_had_pz,  w )
    histograms['reso_b_had_pt'].Fill(  reso_b_had_pt,  w )
    histograms['reso_b_had_y'].Fill(   reso_b_had_y,   w )
    histograms['reso_b_had_phi'].Fill( reso_b_had_phi, w )
    histograms['reso_b_had_E'].Fill(   reso_b_had_E,   w )
    histograms['reso_b_had_m'].Fill(   reso_b_had_m,   w )

    histograms['reso_t_had_px'].Fill(  reso_t_had_px,  w )
    histograms['reso_t_had_py'].Fill(  reso_t_had_py,  w )
    histograms['reso_t_had_pz'].Fill(  reso_t_had_pz,  w )
    histograms['reso_t_had_pt'].Fill(  reso_t_had_pt,  w )
    histograms['reso_t_had_y'].Fill(   reso_t_had_y,   w )
    histograms['reso_t_had_phi'].Fill( reso_t_had_phi, w )
    histograms['reso_t_had_E'].Fill(   reso_t_had_E,   w )
    histograms['reso_t_had_m'].Fill(   reso_t_had_m,   w )
    
    histograms['reso_W_lep_px'].Fill(  reso_W_lep_px,  w )
    histograms['reso_W_lep_py'].Fill(  reso_W_lep_py,  w )
    histograms['reso_W_lep_pz'].Fill(  reso_W_lep_pz,  w )
    histograms['reso_W_lep_pt'].Fill(  reso_W_lep_pt,  w )
    histograms['reso_W_lep_y'].Fill(   reso_W_lep_y, w )
    histograms['reso_W_lep_phi'].Fill( reso_W_lep_phi, w )
    histograms['reso_W_lep_E'].Fill(   reso_W_lep_E,   w )
    histograms['reso_W_lep_m'].Fill(   reso_W_lep_m,   w )

    histograms['reso_b_lep_px'].Fill(  reso_b_lep_px,  w )
    histograms['reso_b_lep_py'].Fill(  reso_b_lep_py,  w )
    histograms['reso_b_lep_pz'].Fill(  reso_b_lep_pz,  w )
    histograms['reso_b_lep_pt'].Fill(  reso_b_lep_pt,  w )
    histograms['reso_b_lep_y'].Fill(   reso_b_lep_y, w )
    histograms['reso_b_lep_phi'].Fill( reso_b_lep_phi, w )
    histograms['reso_b_lep_E'].Fill(   reso_b_lep_E,   w )
    histograms['reso_b_lep_m'].Fill(   reso_b_lep_m,   w )

    histograms['reso_t_lep_px'].Fill(  reso_t_lep_px,  w )
    histograms['reso_t_lep_py'].Fill(  reso_t_lep_py,  w )
    histograms['reso_t_lep_pz'].Fill(  reso_t_lep_pz,  w )
    histograms['reso_t_lep_pt'].Fill(  reso_t_lep_pt,  w )
    histograms['reso_t_lep_y'].Fill(   reso_t_lep_y, w )
    histograms['reso_t_lep_phi'].Fill( reso_t_lep_phi, w )
    histograms['reso_t_lep_E'].Fill(   reso_t_lep_E,   w )
    histograms['reso_t_lep_m'].Fill(   reso_t_lep_m,   w )

    n_good += 1
    
    if i < 10:
       print "rn=%-10i en=%-10i ) Hadronic top      :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
               ( event_info[i][0], event_info[i][1],
                t_had_true.Pt(),   t_had_true.Rapidity(),   t_had_true.Phi(),   t_had_true.E(),   t_had_true.M(), \
                t_had_fitted.Pt(), t_had_fitted.Rapidity(), t_had_fitted.Phi(), t_had_fitted.E(), t_had_fitted.M() )
   
       print "rn=%-10i en=%-10i ) Leptonic top  :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
               ( event_info[i][0], event_info[i][1],
                t_lep_true.Pt(),   t_lep_true.Rapidity(),   t_lep_true.Phi(),   t_lep_true.E(),   t_lep_true.M(), \
                t_lep_fitted.Pt(), t_lep_fitted.Rapidity(), t_lep_fitted.Phi(), t_lep_fitted.E(), t_lep_fitted.M() )
    
ofile.Write()
ofile.Close()

print "Finished. Saved output file:", ofilename

f_good = 100. * float( n_good ) / float( n_events )
print "Good events: %.2f" % f_good
