#!/usr/bin/env python

GeV = 1e3
TeV = 1e6
Mtop = 172.5

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

def MakeTopsP4( y ):
  t_lep = TLorentzVector()
  t_had = TLorentzVector()

  px_lep = y[0]
  py_lep = y[1]
  pz_lep = y[2]
  px_had = y[3]
  py_had = y[4]
  pz_had = y[5]

  # E^2 - P^2 = M^2 => E^2 = P^2 + M^2
  P_lep = TMath.Sqrt( px_lep*px_lep + py_lep*py_lep + pz_lep*pz_lep )
  E_lep = TMath.Sqrt( P_lep*P_lep + Mtop*Mtop )
  
  P_had = TMath.Sqrt( px_had*px_had + py_had*py_had + pz_had*pz_had )
  E_had = TMath.Sqrt( P_had*P_had + Mtop*Mtop )

  t_lep.SetPxPyPzE( px_lep, py_lep, pz_lep, E_lep )
  t_had.SetPxPyPzE( px_had, py_had, pz_had, E_had )

  return t_lep, t_had

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

input_features_t_lep  = input_features_t_lep
input_features_t_had  = input_features_t_had
models.n_cols_t_lep   = n_features_per_jet
models.n_rows_t_lep   = n_rows_t_lep
models.n_cols_t_had   = n_features_per_jet
models.n_rows_t_had   = n_rows_t_had

target_features          = target_features_ttbar
models.n_target_features = len(target_features)

#print "INFO: input features:"
#print input_features

X_t_lep = data[input_features_t_lep].values
X_t_had = data[input_features_t_had].values
y_true  = data[target_features].values

event_info = data[features_event_info].values

n_events = len(data)

#X_t_lep = X_t_lep_scaler.transform(X_t_lep)
#X_t_had = X_t_had_scaler.transform(X_t_had)

X_t_lep = X_t_lep.reshape( n_events, models.n_rows_t_lep, models.n_cols_t_lep )
X_t_had = X_t_had.reshape( n_events, models.n_rows_t_had, models.n_cols_t_had )


y_fitted = dnn.predict( [ X_t_lep, X_t_had ] )
#y_fitted = y_scaler.inverse_transform( y_fitted )

# open output file
ofilename = "output/testing.root"
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

#################
# book histograms
histograms = {}

# basic distributions
histograms['t_had_pt']       = TH1F( "t_had_pt",  ";Hadronic top p_{T} [GeV]", 50, 0., 1500. )
histograms['t_had_y']        = TH1F( "t_had_y",   ";Hadronic top #eta", 25, -5., 5. )
histograms['t_had_phi']      = TH1F( "t_had_phi", ";Hadronic top #phi", 32, -3.2, 3.2 )
histograms['t_had_E']        = TH1F( "t_had_E",   ";Hadronic top E [GeV]", 50, 0., 1500. )
histograms['t_had_m']        = TH1F( "t_had_m",   ";Hadronic top m [GeV]", 30, 0., 300.  )

histograms['t_lep_pt']       = TH1F( "t_lep_pt",   ";Leptonic top p_{T} [GeV]", 50, 0., 1500. )
histograms['t_lep_y']        = TH1F( "t_lep_y",    ";Leptonic top #eta", 25, -5., 5. )
histograms['t_lep_phi']      = TH1F( "t_lep_phi",  ";Leptonic top #phi", 32, -3.2, 3.2 )
histograms['t_lep_E']        = TH1F( "t_lep_E",    ";Leptonic top E [GeV]", 50, 0., 1500. )
histograms['t_lep_m']        = TH1F( "t_lep_m",    ";Leptonic top m [GeV]", 30, 0., 300. )

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
histograms['reso_t_had_px']   = TH1F( "reso_t_had_px",   ";Hadronic top p_{x} resolution", 100, -3.0, 3.0 )
histograms['reso_t_had_py']   = TH1F( "reso_t_had_py",   ";Hadronic top p_{y} resolution", 100, -3.0, 3.0 )
histograms['reso_t_had_pz']   = TH1F( "reso_t_had_pz",   ";Hadronic top p_{z} resolution", 100, -3.0, 3.0 )
histograms['reso_t_had_pt']   = TH1F( "reso_t_had_pt",   ";Hadronic top p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_t_had_y']    = TH1F( "reso_t_had_y",    ";Hadronic top y resolution",  100, -3.0, 3.0 )
histograms['reso_t_had_phi']  = TH1F( "reso_t_had_phi",  ";Hadronic top #phi resolution",  100, -3.0, 3.0 )
histograms['reso_t_had_E']    = TH1F( "reso_t_had_E",    ";Hadronic top E resolution",     100, -3.0, 3.0 )
histograms['reso_t_had_m']    = TH1F( "reso_t_had_m",    ";Hadronic top M resolution",     100, -3.0, 3.0 )

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

    t_had_true,   t_lep_true   = MakeTopsP4( y_true[i] )
    t_had_fitted, t_lep_fitted = MakeTopsP4( y_fitted[i] )

#    t_had_true    = TLorentzVector()
#    t_had_fitted  = TLorentzVector()
#    t_had_true.SetPxPyPzE(   y_true[i][0], y_true[i][1], y_true[i][2], y_true[i][3] )
#    t_had_fitted.SetPxPyPzE( y_fitted[i][0], y_fitted[i][1], y_fitted[i][2], y_fitted[i][3] )
    
#    t_lep_true    = TLorentzVector()
#    t_lep_fitted  = TLorentzVector()
#    t_lep_true.SetPxPyPzE(   y_true[i][4], y_true[i][5], y_true[i][6], y_true[i][7] )
#    t_lep_fitted.SetPxPyPzE( y_fitted[i][4], y_fitted[i][5], y_fitted[i][6], y_fitted[i][7] )

    try:
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

#    histograms['t_had_px'].Fill(  t_had_fitted.Px(),  w )
    histograms['t_had_pt'].Fill(  t_had_fitted.Pt(),  w )
    histograms['t_had_y'].Fill(   t_had_fitted.Rapidity(), w )
    histograms['t_had_phi'].Fill( t_had_fitted.Phi(), w )
    histograms['t_had_E'].Fill(   t_had_fitted.E(),   w )
    histograms['t_had_m'].Fill(   t_had_fitted.M(),   w )
    
    histograms['t_lep_pt'].Fill(  t_lep_fitted.Pt(),  w )
    histograms['t_lep_y'].Fill(   t_lep_fitted.Rapidity(), w )
    histograms['t_lep_phi'].Fill( t_lep_fitted.Phi(), w )
    histograms['t_lep_E'].Fill(   t_lep_fitted.E(),   w )
    histograms['t_lep_m'].Fill(   t_lep_fitted.M(),   w )

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

    histograms['reso_t_had_px'].Fill(  reso_t_had_px,  w )
    histograms['reso_t_had_py'].Fill(  reso_t_had_py,  w )
    histograms['reso_t_had_pz'].Fill(  reso_t_had_pz,  w )
    histograms['reso_t_had_pt'].Fill(  reso_t_had_pt,  w )
    histograms['reso_t_had_y'].Fill(   reso_t_had_y,   w )
    histograms['reso_t_had_phi'].Fill( reso_t_had_phi, w )
    histograms['reso_t_had_E'].Fill(   reso_t_had_E,   w )
    histograms['reso_t_had_m'].Fill(   reso_t_had_m,   w )
    
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
