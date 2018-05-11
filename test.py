#!/usr/bin/env python

GeV = 1e3
TeV = 1e6

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
  X_scaler = pickle.load( file_scaler )
  y_scaler = pickle.load( file_scaler )

# read in input file
data = pd.read_csv( infilename, delimiter=',', names=header )

print "INFO: input features:"
print input_features
X = data[input_features].values
print "INFO: input four momenta:"
print X

y = data[target_features].values
print "INFO: target features:"
print target_features
print "INFO: target top and antitop four momenta:"
print y

event_info = data[['runNumber','eventNumber','weight']].values

n_events = len(data)

X_scaled = X_scaler.transform(X)
y_scaled = y_scaler.transform(y)

X_scaled = X_scaled.reshape( ( n_events, (1+n_jets_per_event), n_features_per_jet) )
print "INFO: input shape:", X_scaled.shape
print "INFO: target shape:", y_scaled.shape

y_fitted = dnn.predict( X_scaled )
y_fitted = y_scaler.inverse_transform( y_fitted )

# open output file
ofilename = "output/testing.root"
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

# book histograms
histograms = {}
histograms['t_pt']       = TH1F( "t_pt", ";Top p_{T} [GeV]", 50, 0., 1500. )
histograms['t_y']      = TH1F( "t_y", ";Top #eta", 25, -5., 5. )
histograms['t_phi']      = TH1F( "t_phi", ";Top #phi", 32, -3.2, 3.2 )
histograms['t_E']        = TH1F( "t_E",  ";Top E [GeV]", 50, 0., 1500. )
histograms['t_m']        = TH1F( "t_m", ";Top m [GeV]", 30, 0., 300.  )
histograms['tb_pt']       = TH1F( "tb_pt", ";Anti-top p_{T} [GeV]", 50, 0., 1500. )
histograms['tb_y']      = TH1F( "tb_y", ";Anti-top #eta", 25, -5., 5. )
histograms['tb_phi']      = TH1F( "tb_phi", ";Anti-top #phi", 32, -3.2, 3.2 )
histograms['tb_E']        = TH1F( "tb_E",  ";Anti-top E [GeV]", 50, 0., 1500. )
histograms['tb_m']        = TH1F( "tb_m", ";Anti-top m [GeV]", 30, 0., 300. )

histograms['corr_t_pt']    = TH2F( "corr_t_pt", ";True Top p_{T} [GeV];Fitted Top p_{T} [GeV]", 50, 0., 1500., 50, 0., 1500. )
histograms['corr_t_y']     = TH2F( "corr_t_y",       ";True Top y;Fitted Top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_t_phi']   = TH2F( "corr_t_phi",     ";True Top #phi;Fitted Top #phi", 32, -3.2, 3.2, 32, -3.2, 3.2 )
histograms['corr_t_E']     = TH2F( "corr_t_E",       ";True Top E [GeV];Fitted Top E [GeV]", 50, 0., 1500., 50, 0., 1500. )
histograms['corr_t_m']     = TH2F( "corr_t_m",       ";True Top m [GeV];Fitted Top m [GeV]", 30, 0., 300., 30, 0., 300. )
histograms['corr_tb_pt']   = TH2F( "corr_tb_pt",     ";True Anti-top p_{T} [GeV];Fitted Anti-top p_{T} [GeV]", 50, 0., 1500., 50, 0., 1500. )
histograms['corr_tb_y']    = TH2F( "corr_tb_y",      ";True Anti-top y;Fitted Anti-top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_tb_phi']  = TH2F( "corr_tb_phi",    ";True Anti-top #phi;Fitted Anti-top #phi", 32, -3.2, 3.2, 32, -3.2, 3.2 )
histograms['corr_tb_E']    = TH2F( "corr_tb_E",      ";True Anti-top E [GeV];Fitted Anti-top E [GeV]", 50, 0., 1500., 50, 0., 1500. )
histograms['corr_tb_m']    = TH2F( "corr_tb_m",      ";True Anti-top m [GeV];Fitted Anti-top m [GeV]", 30, 0., 300., 30, 0., 300. )
    
histograms['reso_t_pt']   = TH1F( "reso_t_pt",   "Top p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_t_y']    = TH1F( "reso_t_y",    "Top y resolution",  100, -3.0, 3.0 )
histograms['reso_t_phi']  = TH1F( "reso_t_phi",  "Top #phi resolution",  100, -3.0, 3.0 )
histograms['reso_t_E']    = TH1F( "reso_t_E",    "Top E resolution",     100, -3.0, 3.0 )
histograms['reso_t_m']    = TH1F( "reso_t_m",    "Top M resolution",     100, -3.0, 3.0 )
histograms['reso_tb_pt']  = TH1F( "reso_tb_pt",  "Anti-top p_{T} resolution", 100, -3.0, 3.0 )
histograms['reso_tb_y']   = TH1F( "reso_tb_y",   "Anti-top y resolution",  100, -3.0, 3.0 )
histograms['reso_tb_phi'] = TH1F( "reso_tb_phi", "Anti-top #phi resolution",  100, -3.0, 3.0 )
histograms['reso_tb_E']   = TH1F( "reso_tb_E",   "Anti-top E resolution",     100, -3.0, 3.0 )
histograms['reso_tb_m']   = TH1F( "reso_tb_m",   "Anti-top M resolution",     100, -3.0, 3.0 )
for h in histograms.values(): h.Sumw2()

print "INFO: starting event loop. Found %i events" % n_events
n_good = 0
# Print out example
for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print "INFO: Event %-9i  (%3.0f %%)" % ( i, perc )

    t_true    = TLorentzVector()
    tb_true   = TLorentzVector()
    t_fitted  = TLorentzVector()
    tb_fitted = TLorentzVector()

    w = event_info[i][2]

    t_true.SetPxPyPzE(  y[i][0], y[i][1], y[i][2], y[i][3] )
    tb_true.SetPxPyPzE( y[i][4], y[i][5], y[i][6], y[i][7] )

    t_fitted.SetPxPyPzE(  y_fitted[i][0], y_fitted[i][1], y_fitted[i][2], y_fitted[i][3] )
    tb_fitted.SetPxPyPzE( y_fitted[i][4], y_fitted[i][5], y_fitted[i][6], y_fitted[i][7] )

    try:
        reso_t_pt  = ( t_fitted.Pt()  - t_true.Pt()  ) / t_true.Pt()   #if t_true.Pt()  != 0. else -1000.
        reso_t_y   = ( t_fitted.Rapidity() - t_true.Rapidity() ) / t_true.Rapidity()  #if t_true.Rapidity() != 0. else -1000.
        reso_t_phi = ( t_fitted.Phi() - t_true.Phi() ) / t_true.Phi()  #if t_true.Phi() != 0. else -1000.
        reso_t_E   = ( t_fitted.E()   - t_true.E()   ) / t_true.E()    #if t_true.E()   != 0. else -1000.
        reso_t_m   = ( t_fitted.M()   - t_true.M()   ) / t_true.M()    #if t_true.M()   != 0. else -1000.
       
        reso_tb_pt  = ( tb_fitted.Pt()  - tb_true.Pt()  ) / tb_true.Pt()    #if tb_true.Pt()   != 0. else -1000.
        reso_tb_y = ( tb_fitted.Rapidity() - tb_true.Rapidity() ) / tb_true.Rapidity()   #if tb_true.Rapidity()  != 0. else -1000.
        reso_tb_phi = ( tb_fitted.Phi() - tb_true.Phi() ) / tb_true.Phi()   #if tb_true.Phi()  != 0. else -1000.
        reso_tb_E   = ( tb_fitted.E()   - tb_true.E()   ) / tb_true.E()     #if tb_true.E()    != 0. else -1000.
        reso_tb_m   = ( tb_fitted.M()   - tb_true.M()   ) / tb_true.M()     #if tb_true.M()    != 0. else -1000.
    except:
        print "WARNING: invalid tops, skipping event ( rn=%-10i en=%-10i )" % ( event_info[i][0], event_info[i][1] )
        continue

    histograms['t_pt'].Fill(  t_fitted.Pt(),  w )
    histograms['t_y'].Fill( t_fitted.Rapidity(), w )
    histograms['t_phi'].Fill( t_fitted.Phi(), w )
    histograms['t_E'].Fill(   t_fitted.E(),   w )
    histograms['t_m'].Fill(   t_fitted.M(),   w )
    histograms['tb_pt'].Fill(  tb_fitted.Pt(),  w )
    histograms['tb_y'].Fill( tb_fitted.Rapidity(), w )
    histograms['tb_phi'].Fill( tb_fitted.Phi(), w )
    histograms['tb_E'].Fill(   tb_fitted.E(),   w )
    histograms['tb_m'].Fill(   tb_fitted.M(),   w )

    histograms['corr_t_pt'].Fill(   t_true.Pt(),       t_fitted.Pt(),  w )
    histograms['corr_t_y'].Fill(    t_true.Rapidity(), t_fitted.Rapidity(), w )
    histograms['corr_t_phi'].Fill(  t_true.Phi(),      t_fitted.Phi(), w )
    histograms['corr_t_E'].Fill(    t_true.E(),        t_fitted.E(),   w )
    histograms['corr_t_m'].Fill(    t_true.M(),        t_fitted.M(),   w )
    histograms['corr_tb_pt'].Fill(  t_true.Pt(),       tb_fitted.Pt(),  w )
    histograms['corr_tb_y'].Fill(   t_true.Rapidity(), tb_fitted.Rapidity(), w )
    histograms['corr_tb_phi'].Fill( t_true.Phi(),      tb_fitted.Phi(), w )
    histograms['corr_tb_E'].Fill(   t_true.E(),        tb_fitted.E(),   w )
    histograms['corr_tb_m'].Fill(   t_true.M(),        tb_fitted.M(),   w )
    
    histograms['reso_t_pt'].Fill(  reso_t_pt,  w )
    histograms['reso_t_y'].Fill( reso_t_y, w )
    histograms['reso_t_phi'].Fill( reso_t_phi, w )
    histograms['reso_t_E'].Fill(   reso_t_E,   w )
    histograms['reso_t_m'].Fill(   reso_t_m,   w )
    histograms['reso_tb_pt'].Fill(  reso_tb_pt,  w )
    histograms['reso_tb_y'].Fill( reso_tb_y, w )
    histograms['reso_tb_phi'].Fill( reso_tb_phi, w )
    histograms['reso_tb_E'].Fill(   reso_tb_E,   w )
    histograms['reso_tb_m'].Fill(   reso_tb_m,   w )

    n_good += 1
    
    if i < 10:
       print "rn=%-10i en=%-10i ) Top      :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
               ( event_info[i][0], event_info[i][1],
                t_true.Pt(),   t_true.Rapidity(),   t_true.Phi(),   t_true.E(),   t_true.M(), \
                t_fitted.Pt(), t_fitted.Rapidity(), t_fitted.Phi(), t_fitted.E(), t_fitted.M() )
   
       print "rn=%-10i en=%-10i ) AntiTop  :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
               ( event_info[i][0], event_info[i][1],
                tb_true.Pt(),   tb_true.Rapidity(),   tb_true.Phi(),   tb_true.E(),   tb_true.M(), \
                tb_fitted.Pt(), tb_fitted.Rapidity(), tb_fitted.Phi(), tb_fitted.E(), tb_fitted.M() )
    
ofile.Write()
ofile.Close()

print "Finished. Saved output file:", ofilename

f_good = 100. * float( n_good ) / float( n_events )
print "Good events: %.2f" % f_good
