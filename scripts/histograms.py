#!/usr/bin/env python

GeV = 1e3
TeV = 1e6
m_t = 172.5
m_W = 80.4
m_b = 4.95

import os, sys, time
import argparse

from ROOT import *
from array import array
import cPickle as pickle


################

def PrintOut( p4_true, p4_fitted, label ):
  print "%s :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
               ( label,
                p4_true.Pt(),   p4_true.Rapidity(),   p4_true.Phi(),   p4_true.E(),   p4_true.M(), \
                p4_fitted.Pt(), p4_fitted.Rapidity(), p4_fitted.Phi(), p4_fitted.E(), p4_fitted.M() )

###############

infilename = "output/fitted.root"
if len(sys.argv) > 1: infilename = sys.argv[1]



# read in input file
infile = TFile.Open( infilename )
tree   = infile.Get( "nominal") 

# open output file
ofilename = "output/histograms.root"
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

#################
# book histograms
histograms = {}

# basic distributions

# True
histograms['W_had_px_true']       = TH1F( "W_had_px_true",  ";Hadronic W p_{x} [GeV]", 50, -1000., 1000. )
histograms['W_had_py_true']       = TH1F( "W_had_py_true",  ";Hadronic W p_{y} [GeV]", 50, -1000., 1000. )
histograms['W_had_pz_true']       = TH1F( "W_had_pz_true",  ";Hadronic W p_{z} [GeV]", 50, -1000., 1000. )
histograms['W_had_pt_true']       = TH1F( "W_had_pt_true",  ";Hadronic W p_{T} [GeV]", 50, 0., 500. )
histograms['W_had_y_true']        = TH1F( "W_had_y_true",   ";Hadronic W #eta", 25, -5., 5. )
histograms['W_had_phi_true']      = TH1F( "W_had_phi_true", ";Hadronic W #phi", 16, -3.2, 3.2 )
histograms['W_had_E_true']        = TH1F( "W_had_E_true",   ";Hadronic W E [GeV]", 50, 0., 500. )
histograms['W_had_m_true']        = TH1F( "W_had_m_true",   ";Hadronic W m [GeV]", 30, 0., 300.  )

histograms['b_had_px_true']       = TH1F( "b_had_px_true",  ";Hadronic b p_{x} [GeV]", 50, -1000., 1000. )
histograms['b_had_py_true']       = TH1F( "b_had_py_true",  ";Hadronic b p_{y} [GeV]", 50, -1000., 1000. )
histograms['b_had_pz_true']       = TH1F( "b_had_pz_true",  ";Hadronic b p_{z} [GeV]", 50, -1000., 1000. )
histograms['b_had_pt_true']       = TH1F( "b_had_pt_true",  ";Hadronic b p_{T} [GeV]", 50, 0., 500. )
histograms['b_had_y_true']        = TH1F( "b_had_y_true",   ";Hadronic b #eta", 25, -5., 5. )
histograms['b_had_phi_true']      = TH1F( "b_had_phi_true", ";Hadronic b #phi", 16, -3.2, 3.2 )
histograms['b_had_E_true']        = TH1F( "b_had_E_true",   ";Hadronic b E [GeV]", 50, 0., 500. )
histograms['b_had_m_true']        = TH1F( "b_had_m_true",   ";Hadronic b m [GeV]", 30, 0., 300.  )

histograms['t_had_px_true']       = TH1F( "t_had_px_true",  ";Hadronic t p_{x} [GeV]", 50, -1000., 1000. )
histograms['t_had_py_true']       = TH1F( "t_had_py_true",  ";Hadronic t p_{y} [GeV]", 50, -1000., 1000. )
histograms['t_had_pz_true']       = TH1F( "t_had_pz_true",  ";Hadronic t p_{z} [GeV]", 50, -1000., 1000. )
histograms['t_had_pt_true']       = TH1F( "t_had_pt_true",  ";Hadronic t p_{T} [GeV]", 50, 0., 500. )
histograms['t_had_y_true']        = TH1F( "t_had_y_true",   ";Hadronic t #eta", 25, -5., 5. )
histograms['t_had_phi_true']      = TH1F( "t_had_phi_true", ";Hadronic t #phi", 16, -3.2, 3.2 )
histograms['t_had_E_true']        = TH1F( "t_had_E_true",   ";Hadronic t E [GeV]", 50, 0., 500. )
histograms['t_had_m_true']        = TH1F( "t_had_m_true",   ";Hadronic t m [GeV]", 30, 0., 300.  )

histograms['W_lep_px_true']       = TH1F( "W_lep_px_true",   ";Leptonic W p_{x} [GeV]", 50, -1000., 1000. )
histograms['W_lep_py_true']       = TH1F( "W_lep_py_true",   ";Leptonic W p_{y} [GeV]", 50, -1000., 1000. )
histograms['W_lep_pz_true']       = TH1F( "W_lep_pz_true",   ";Leptonic W p_{z} [GeV]", 50, -1000., 1000. )
histograms['W_lep_pt_true']       = TH1F( "W_lep_pt_true",   ";Leptonic W p_{T} [GeV]", 50, 0., 500. )
histograms['W_lep_y_true']        = TH1F( "W_lep_y_true",    ";Leptonic W #eta", 25, -5., 5. )
histograms['W_lep_phi_true']      = TH1F( "W_lep_phi_true",  ";Leptonic W #phi", 16, -3.2, 3.2 )
histograms['W_lep_E_true']        = TH1F( "W_lep_E_true",    ";Leptonic W E [GeV]", 50, 0., 500. )
histograms['W_lep_m_true']        = TH1F( "W_lep_m_true",    ";Leptonic W m [GeV]", 30, 0., 300. )

histograms['b_lep_px_true']       = TH1F( "b_lep_px_true",   ";Leptonic b p_{x} [GeV]", 50, -1000., 1000. )
histograms['b_lep_py_true']       = TH1F( "b_lep_py_true",   ";Leptonic b p_{y} [GeV]", 50, -1000., 1000. )
histograms['b_lep_pz_true']       = TH1F( "b_lep_pz_true",   ";Leptonic b p_{z} [GeV]", 50, -1000., 1000. )
histograms['b_lep_pt_true']       = TH1F( "b_lep_pt_true",   ";Leptonic b p_{T} [GeV]", 50, 0., 500. )
histograms['b_lep_y_true']        = TH1F( "b_lep_y_true",    ";Leptonic b #eta", 25, -5., 5. )
histograms['b_lep_phi_true']      = TH1F( "b_lep_phi_true",  ";Leptonic b #phi", 16, -3.2, 3.2 )
histograms['b_lep_E_true']        = TH1F( "b_lep_E_true",    ";Leptonic b E [GeV]", 50, 0., 500. )
histograms['b_lep_m_true']        = TH1F( "b_lep_m_true",    ";Leptonic b m [GeV]", 30, 0., 300. )

histograms['t_lep_px_true']       = TH1F( "t_lep_px_true",   ";Leptonic t p_{x} [GeV]", 50, -1000., 1000. )
histograms['t_lep_py_true']       = TH1F( "t_lep_py_true",   ";Leptonic t p_{y} [GeV]", 50, -1000., 1000. )
histograms['t_lep_pz_true']       = TH1F( "t_lep_pz_true",   ";Leptonic t p_{z} [GeV]", 50, -1000., 1000. )
histograms['t_lep_pt_true']       = TH1F( "t_lep_pt_true",   ";Leptonic t p_{T} [GeV]", 50, 0., 500. )
histograms['t_lep_y_true']        = TH1F( "t_lep_y_true",    ";Leptonic t #eta", 25, -5., 5. )
histograms['t_lep_phi_true']      = TH1F( "t_lep_phi_true",  ";Leptonic t #phi", 16, -3.2, 3.2 )
histograms['t_lep_E_true']        = TH1F( "t_lep_E_true",    ";Leptonic t E [GeV]", 50, 0., 500. )
histograms['t_lep_m_true']        = TH1F( "t_lep_m_true",    ";Leptonic t m [GeV]", 30, 0., 300. )

# Fitted
histograms['W_had_px_fitted']       = TH1F( "W_had_px_fitted",  ";Hadronic W p_{x} [GeV]", 50, -1000., 1000. )
histograms['W_had_py_fitted']       = TH1F( "W_had_py_fitted",  ";Hadronic W p_{y} [GeV]", 50, -1000., 1000. )
histograms['W_had_pz_fitted']       = TH1F( "W_had_pz_fitted",  ";Hadronic W p_{z} [GeV]", 50, -1000., 1000. )
histograms['W_had_pt_fitted']       = TH1F( "W_had_pt_fitted",  ";Hadronic W p_{T} [GeV]", 50, 0., 500. )
histograms['W_had_y_fitted']        = TH1F( "W_had_y_fitted",   ";Hadronic W #eta", 25, -5., 5. )
histograms['W_had_phi_fitted']      = TH1F( "W_had_phi_fitted", ";Hadronic W #phi", 16, -3.2, 3.2 )
histograms['W_had_E_fitted']        = TH1F( "W_had_E_fitted",   ";Hadronic W E [GeV]", 50, 0., 500. )
histograms['W_had_m_fitted']        = TH1F( "W_had_m_fitted",   ";Hadronic W m [GeV]", 30, 0., 300.  )

histograms['b_had_px_fitted']       = TH1F( "b_had_px_fitted",  ";Hadronic b p_{x} [GeV]", 50, -1000., 1000. )
histograms['b_had_py_fitted']       = TH1F( "b_had_py_fitted",  ";Hadronic b p_{y} [GeV]", 50, -1000., 1000. )
histograms['b_had_pz_fitted']       = TH1F( "b_had_pz_fitted",  ";Hadronic b p_{z} [GeV]", 50, -1000., 1000. )
histograms['b_had_pt_fitted']       = TH1F( "b_had_pt_fitted",  ";Hadronic b p_{T} [GeV]", 50, 0., 500. )
histograms['b_had_y_fitted']        = TH1F( "b_had_y_fitted",   ";Hadronic b #eta", 25, -5., 5. )
histograms['b_had_phi_fitted']      = TH1F( "b_had_phi_fitted", ";Hadronic b #phi", 16, -3.2, 3.2 )
histograms['b_had_E_fitted']        = TH1F( "b_had_E_fitted",   ";Hadronic b E [GeV]", 50, 0., 500. )
histograms['b_had_m_fitted']        = TH1F( "b_had_m_fitted",   ";Hadronic b m [GeV]", 30, 0., 300.  )

histograms['t_had_px_fitted']       = TH1F( "t_had_px_fitted",  ";Hadronic t p_{x} [GeV]", 50, -1000., 1000. )
histograms['t_had_py_fitted']       = TH1F( "t_had_py_fitted",  ";Hadronic t p_{y} [GeV]", 50, -1000., 1000. )
histograms['t_had_pz_fitted']       = TH1F( "t_had_pz_fitted",  ";Hadronic t p_{z} [GeV]", 50, -1000., 1000. )
histograms['t_had_pt_fitted']       = TH1F( "t_had_pt_fitted",  ";Hadronic top p_{T} [GeV]", 50, 0., 500. )
histograms['t_had_y_fitted']        = TH1F( "t_had_y_fitted",   ";Hadronic top #eta", 25, -5., 5. )
histograms['t_had_phi_fitted']      = TH1F( "t_had_phi_fitted", ";Hadronic top #phi", 16, -3.2, 3.2 )
histograms['t_had_E_fitted']        = TH1F( "t_had_E_fitted",   ";Hadronic top E [GeV]", 50, 0., 500. )
histograms['t_had_m_fitted']        = TH1F( "t_had_m_fitted",   ";Hadronic top m [GeV]", 30, 0., 300.  )

histograms['W_lep_px_fitted']       = TH1F( "W_lep_px_fitted",   ";Leptonic W p_{x} [GeV]", 50, -1000., 1000. )
histograms['W_lep_py_fitted']       = TH1F( "W_lep_py_fitted",   ";Leptonic W p_{y} [GeV]", 50, -1000., 1000. )
histograms['W_lep_pz_fitted']       = TH1F( "W_lep_pz_fitted",   ";Leptonic W p_{z} [GeV]", 50, -1000., 1000. )
histograms['W_lep_pt_fitted']       = TH1F( "W_lep_pt_fitted",   ";Leptonic W p_{T} [GeV]", 50, 0., 500. )
histograms['W_lep_y_fitted']        = TH1F( "W_lep_y_fitted",    ";Leptonic W #eta", 25, -5., 5. )
histograms['W_lep_phi_fitted']      = TH1F( "W_lep_phi_fitted",  ";Leptonic W #phi", 16, -3.2, 3.2 )
histograms['W_lep_E_fitted']        = TH1F( "W_lep_E_fitted",    ";Leptonic W E [GeV]", 50, 0., 500. )
histograms['W_lep_m_fitted']        = TH1F( "W_lep_m_fitted",    ";Leptonic W m [GeV]", 30, 0., 300. )

histograms['b_lep_px_fitted']       = TH1F( "b_lep_px_fitted",   ";Leptonic b p_{x} [GeV]", 50, -1000., 1000. )
histograms['b_lep_py_fitted']       = TH1F( "b_lep_py_fitted",   ";Leptonic b p_{y} [GeV]", 50, -1000., 1000. )
histograms['b_lep_pz_fitted']       = TH1F( "b_lep_pz_fitted",   ";Leptonic b p_{z} [GeV]", 50, -1000., 1000. )
histograms['b_lep_pt_fitted']       = TH1F( "b_lep_pt_fitted",   ";Leptonic b p_{T} [GeV]", 50, 0., 500. )
histograms['b_lep_y_fitted']        = TH1F( "b_lep_y_fitted",    ";Leptonic b #eta", 25, -5., 5. )
histograms['b_lep_phi_fitted']      = TH1F( "b_lep_phi_fitted",  ";Leptonic b #phi", 16, -3.2, 3.2 )
histograms['b_lep_E_fitted']        = TH1F( "b_lep_E_fitted",    ";Leptonic b E [GeV]", 50, 0., 500. )
histograms['b_lep_m_fitted']        = TH1F( "b_lep_m_fitted",    ";Leptonic b m [GeV]", 30, 0., 300. )

histograms['t_lep_px_fitted']       = TH1F( "t_lep_px_fitted",   ";Leptonic t p_{x} [GeV]", 50, -1000., 1000. )
histograms['t_lep_py_fitted']       = TH1F( "t_lep_py_fitted",   ";Leptonic t p_{y} [GeV]", 50, -1000., 1000. )
histograms['t_lep_pz_fitted']       = TH1F( "t_lep_pz_fitted",   ";Leptonic t p_{z} [GeV]", 50, -1000., 1000. )
histograms['t_lep_pt_fitted']       = TH1F( "t_lep_pt_fitted",   ";Leptonic top p_{T} [GeV]", 50, 0., 500. )
histograms['t_lep_y_fitted']        = TH1F( "t_lep_y_fitted",    ";Leptonic top #eta", 25, -5., 5. )
histograms['t_lep_phi_fitted']      = TH1F( "t_lep_phi_fitted",  ";Leptonic top #phi", 16, -3.2, 3.2 )
histograms['t_lep_E_fitted']        = TH1F( "t_lep_E_fitted",    ";Leptonic top E [GeV]", 50, 0., 500. )
histograms['t_lep_m_fitted']        = TH1F( "t_lep_m_fitted",    ";Leptonic top m [GeV]", 30, 0., 300. )

# 2D correlations 
histograms['corr_t_had_pt']    = TH2F( "corr_t_had_pt",      ";True Hadronic top p_{T} [GeV];Fitted Hadronic top p_{T} [GeV]", 50, 0., 500., 50, 0., 500. )
histograms['corr_t_had_y']     = TH2F( "corr_t_had_y",       ";True Hadronic top y;Fitted Hadronic top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_t_had_phi']   = TH2F( "corr_t_had_phi",     ";True Hadronic top #phi;Fitted Hadronic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
histograms['corr_t_had_E']     = TH2F( "corr_t_had_E",       ";True Hadronic top E [GeV];Fitted Hadronic top E [GeV]", 50, 0., 500., 50, 0., 500. )
histograms['corr_t_had_m']     = TH2F( "corr_t_had_m",       ";True Hadronic top m [GeV];Fitted Hadronic top m [GeV]", 25, 170., 175., 20, 150., 250. )

histograms['corr_t_lep_pt']    = TH2F( "corr_t_lep_pt",     ";True Leptonic top p_{T} [GeV];Fitted Leptonic top p_{T} [GeV]", 50, 0., 500., 50, 0., 500. )
histograms['corr_t_lep_y']     = TH2F( "corr_t_lep_y",      ";True Leptonic top y;Fitted Leptonic top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_t_lep_phi']   = TH2F( "corr_t_lep_phi",    ";True Leptonic top #phi;Fitted Leptonic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
histograms['corr_t_lep_E']     = TH2F( "corr_t_lep_E",      ";True Leptonic top E [GeV];Fitted Leptonic top E [GeV]", 50, 0., 500., 50, 0., 500. )
histograms['corr_t_lep_m']     = TH2F( "corr_t_lep_m",      ";True Leptonic top m [GeV];Fitted Leptonic top m [GeV]", 25, 170., 175., 20, 150., 250. )

# residuals
histograms['diff_W_had_px']       = TH1F( "diff_W_had_px",  ";Hadronic W p_{x} [GeV]", 50, -500., 500. )
histograms['diff_W_had_py']       = TH1F( "diff_W_had_py",  ";Hadronic W p_{y} [GeV]", 50, -500., 500. )
histograms['diff_W_had_pz']       = TH1F( "diff_W_had_pz",  ";Hadronic W p_{z} [GeV]", 50, -500., 500. )
histograms['diff_W_had_pt']       = TH1F( "diff_W_had_pt",  ";Hadronic W p_{T} [GeV]", 50, -500., 500. )
histograms['diff_W_had_y']        = TH1F( "diff_W_had_y",   ";Hadronic W #eta", 25, -5., 5. )
histograms['diff_W_had_phi']      = TH1F( "diff_W_had_phi", ";Hadronic W #phi", 16, -3.2, 3.2 )
histograms['diff_W_had_E']        = TH1F( "diff_W_had_E",   ";Hadronic W E [GeV]", 50, -500., 500. )
histograms['diff_W_had_m']        = TH1F( "diff_W_had_m",   ";Hadronic W m [GeV]", 30, -300., 300.  )

histograms['diff_b_had_px']       = TH1F( "diff_b_had_px",  ";Hadronic b p_{x} [GeV]", 50, -500., 500. )
histograms['diff_b_had_py']       = TH1F( "diff_b_had_py",  ";Hadronic b p_{y} [GeV]", 50, -500., 500. )
histograms['diff_b_had_pz']       = TH1F( "diff_b_had_pz",  ";Hadronic b p_{z} [GeV]", 50, -500., 500. )
histograms['diff_b_had_pt']       = TH1F( "diff_b_had_pt",  ";Hadronic b p_{T} [GeV]", 50, -500., 500. )
histograms['diff_b_had_y']        = TH1F( "diff_b_had_y",   ";Hadronic b #eta", 25, -5., 5. )
histograms['diff_b_had_phi']      = TH1F( "diff_b_had_phi", ";Hadronic b #phi", 16, -3.2, 3.2 )
histograms['diff_b_had_E']        = TH1F( "diff_b_had_E",   ";Hadronic b E [GeV]", 50, -500., 500. )
histograms['diff_b_had_m']        = TH1F( "diff_b_had_m",   ";Hadronic b m [GeV]", 30, -300., 300.  )

histograms['diff_t_had_px']       = TH1F( "diff_t_had_px",  ";Hadronic t p_{x} [GeV]", 50, -500., 500. )
histograms['diff_t_had_py']       = TH1F( "diff_t_had_py",  ";Hadronic t p_{y} [GeV]", 50, -500., 500. )
histograms['diff_t_had_pz']       = TH1F( "diff_t_had_pz",  ";Hadronic t p_{z} [GeV]", 50, -500., 500. )
histograms['diff_t_had_pt']       = TH1F( "diff_t_had_pt",  ";Hadronic t p_{T} [GeV]", 50, -500., 500. )
histograms['diff_t_had_y']        = TH1F( "diff_t_had_y",   ";Hadronic t #eta", 25, -5., 5. )
histograms['diff_t_had_phi']      = TH1F( "diff_t_had_phi", ";Hadronic t #phi", 16, -3.2, 3.2 )
histograms['diff_t_had_E']        = TH1F( "diff_t_had_E",   ";Hadronic t E [GeV]", 50, -500., 500. )
histograms['diff_t_had_m']        = TH1F( "diff_t_had_m",   ";Hadronic t m [GeV]", 30, -300., 300.  )

histograms['diff_W_lep_px']       = TH1F( "diff_W_lep_px",   ";Leptonic W p_{x} [GeV]", 50, -500., 500. )
histograms['diff_W_lep_py']       = TH1F( "diff_W_lep_py",   ";Leptonic W p_{y} [GeV]", 50, -500., 500. )
histograms['diff_W_lep_pz']       = TH1F( "diff_W_lep_pz",   ";Leptonic W p_{z} [GeV]", 50, -500., 500. )
histograms['diff_W_lep_pt']       = TH1F( "diff_W_lep_pt",   ";Leptonic W p_{T} [GeV]", 50, -500., 500. )
histograms['diff_W_lep_y']        = TH1F( "diff_W_lep_y",    ";Leptonic W #eta", 25, -5., 5. )
histograms['diff_W_lep_phi']      = TH1F( "diff_W_lep_phi",  ";Leptonic W #phi", 16, -3.2, 3.2 )
histograms['diff_W_lep_E']        = TH1F( "diff_W_lep_E",    ";Leptonic W E [GeV]", 50, -500., 500. )
histograms['diff_W_lep_m']        = TH1F( "diff_W_lep_m",    ";Leptonic W m [GeV]", 30, -300., 300. )

histograms['diff_b_lep_px']       = TH1F( "diff_b_lep_px",   ";Leptonic b p_{x} [GeV]", 50, -500., 500. )
histograms['diff_b_lep_py']       = TH1F( "diff_b_lep_py",   ";Leptonic b p_{y} [GeV]", 50, -500., 500. )
histograms['diff_b_lep_pz']       = TH1F( "diff_b_lep_pz",   ";Leptonic b p_{z} [GeV]", 50, -500., 500. )
histograms['diff_b_lep_pt']       = TH1F( "diff_b_lep_pt",   ";Leptonic b p_{T} [GeV]", 50, -500., 500. )
histograms['diff_b_lep_y']        = TH1F( "diff_b_lep_y",    ";Leptonic b #eta", 25, -5., 5. )
histograms['diff_b_lep_phi']      = TH1F( "diff_b_lep_phi",  ";Leptonic b #phi", 16, -3.2, 3.2 )
histograms['diff_b_lep_E']        = TH1F( "diff_b_lep_E",    ";Leptonic b E [GeV]", 50, -500., 500. )
histograms['diff_b_lep_m']        = TH1F( "diff_b_lep_m",    ";Leptonic b m [GeV]", 30, -300., 300. )

histograms['diff_t_lep_px']       = TH1F( "diff_t_lep_px",   ";Leptonic t p_{x} [GeV]", 50, -500., 500. )
histograms['diff_t_lep_py']       = TH1F( "diff_t_lep_py",   ";Leptonic t p_{y} [GeV]", 50, -500., 500. )
histograms['diff_t_lep_pz']       = TH1F( "diff_t_lep_pz",   ";Leptonic t p_{z} [GeV]", 50, -500., 500. )
histograms['diff_t_lep_pt']       = TH1F( "diff_t_lep_pt",   ";Leptonic t p_{T} [GeV]", 50, -500., 500. )
histograms['diff_t_lep_y']        = TH1F( "diff_t_lep_y",    ";Leptonic t #eta", 25, -5., 5. )
histograms['diff_t_lep_phi']      = TH1F( "diff_t_lep_phi",  ";Leptonic t #phi", 16, -3.2, 3.2 )
histograms['diff_t_lep_E']        = TH1F( "diff_t_lep_E",    ";Leptonic t E [GeV]", 50, -500., 500. )
histograms['diff_t_lep_m']        = TH1F( "diff_t_lep_m",    ";Leptonic t m [GeV]", 30, -300., 300. )

# resolution

histograms['reso_W_had_px']   = TH2F( "reso_W_had_px",   ";Hadronic W p_{x} [GeV];Hadronic W p_{x} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_W_had_py']   = TH2F( "reso_W_had_py",   ";Hadronic W p_{y} [GeV];Hadronic W p_{y} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_W_had_pz']   = TH2F( "reso_W_had_pz",   ";Hadronic W p_{z} [GeV];Hadronic W p_{z} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_W_had_pt']   = TH2F( "reso_W_had_pt",   ";Hadronic W p_{T} [GeV];Hadronic W p_{T} resolution", 50, 0., 500., 100, -2.0, 2.0 )
histograms['reso_W_had_y']    = TH2F( "reso_W_had_y",    ";Hadronic W y;Hadronic W y resolution",           25, -5., 5., 100, -2.0, 2.0 )
histograms['reso_W_had_phi']  = TH2F( "reso_W_had_phi",  ";Hadronic W #phi;Hadronic W #phi resolution",        16, -3.2, 3.2, 100, -2.0, 2.0 )
histograms['reso_W_had_E']    = TH2F( "reso_W_had_E",    ";Hadronic W E [GeV];Hadronic W E resolution",     50, 0., 500., 100, -2.0, 2.0 )
histograms['reso_W_had_m']    = TH2F( "reso_W_had_m",    ";Hadronic W M [GeV];Hadronic W M resolution",     30, 0., 300., 100, -2.0, 2.0 )

histograms['reso_b_had_px']   = TH2F( "reso_b_had_px",   ";Hadronic b p_{x} [GeV];Hadronic b p_{x} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_b_had_py']   = TH2F( "reso_b_had_py",   ";Hadronic b p_{y} [GeV];Hadronic b p_{y} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_b_had_pz']   = TH2F( "reso_b_had_pz",   ";Hadronic b p_{z} [GeV];Hadronic b p_{z} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_b_had_pt']   = TH2F( "reso_b_had_pt",   ";Hadronic b p_{T} [GeV];Hadronic b p_{T} resolution", 50, 0., 500., 100, -2.0, 2.0 )
histograms['reso_b_had_y']    = TH2F( "reso_b_had_y",    ";Hadronic b y;Hadronic b y resolution",           25, -5., 5., 100, -2.0, 2.0 )
histograms['reso_b_had_phi']  = TH2F( "reso_b_had_phi",  ";Hadronic b #phi;Hadronic b #phi resolution",        16, -3.2, 3.2, 100, -2.0, 2.0 )
histograms['reso_b_had_E']    = TH2F( "reso_b_had_E",    ";Hadronic b E [GeV];Hadronic b E resolution",     50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_b_had_m']    = TH2F( "reso_b_had_m",    ";Hadronic b M [GeV];Hadronic b M resolution",     30, 0., 300., 100, -2.0, 2.0 )

histograms['reso_t_had_px']   = TH2F( "reso_t_had_px",   ";Hadronic top p_{x} [GeV];Hadronic t p_{x} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_t_had_py']   = TH2F( "reso_t_had_py",   ";Hadronic top p_{y} [GeV];Hadronic t p_{y} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_t_had_pz']   = TH2F( "reso_t_had_pz",   ";Hadronic top p_{z} [GeV];Hadronic t p_{z} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_t_had_pt']   = TH2F( "reso_t_had_pt",   ";Hadronic top p_{T} [GeV];Hadronic t p_{T} resolution", 50, 0., 500., 100, -2.0, 2.0 )
histograms['reso_t_had_y']    = TH2F( "reso_t_had_y",    ";Hadronic top y;Hadronic t y resolution",           25, -5., 5., 100, -2.0, 2.0 )
histograms['reso_t_had_phi']  = TH2F( "reso_t_had_phi",  ";Hadronic top #phi;Hadronic t #phi resolution",        16, -3.2, 3.2, 100, -2.0, 2.0 )
histograms['reso_t_had_E']    = TH2F( "reso_t_had_E",    ";Hadronic top E [GeV];Hadronic t E resolution",     50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_t_had_m']    = TH2F( "reso_t_had_m",    ";Hadronic top M [GeV];Hadronic t M resolution",     30, 0., 300., 100, -2.0, 2.0 )

histograms['reso_W_lep_px']  = TH2F( "reso_W_lep_px",  ";Leptonic W p_{x} [GeV];Leptonic W p_{x} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_W_lep_py']  = TH2F( "reso_W_lep_py",  ";Leptonic W p_{y} [GeV];Leptonic W p_{y} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_W_lep_pz']  = TH2F( "reso_W_lep_pz",  ";Leptonic W p_{z} [GeV];Leptonic W p_{z} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_W_lep_pt']  = TH2F( "reso_W_lep_pt",  ";Leptonic W p_{T} [GeV];Leptonic W p_{T} resolution", 50, 0., 500., 100, -2.0, 2.0 )
histograms['reso_W_lep_y']   = TH2F( "reso_W_lep_y",   ";Leptonic W y;Leptonic W y resolution",           25, -5., 5., 100, -2.0, 2.0 )
histograms['reso_W_lep_phi'] = TH2F( "reso_W_lep_phi", ";Leptonic W #phi;Leptonic W #phi resolution",        16, -3.2, 3.2, 100, -2.0, 2.0 )
histograms['reso_W_lep_E']   = TH2F( "reso_W_lep_E",   ";Leptonic W E [GeV];Leptonic W E resolution",     50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_W_lep_m']   = TH2F( "reso_W_lep_m",   ";Leptonic W M [GeV];Leptonic W M resolution",     30, 0., 300., 100, -2.0, 2.0 )

histograms['reso_b_lep_px']  = TH2F( "reso_b_lep_px",  ";Leptonic b p_{x} [GeV];Leptonic b p_{x} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_b_lep_py']  = TH2F( "reso_b_lep_py",  ";Leptonic b p_{y} [GeV];Leptonic b p_{y} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_b_lep_pz']  = TH2F( "reso_b_lep_pz",  ";Leptonic b p_{z} [GeV];Leptonic b p_{z} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_b_lep_pt']  = TH2F( "reso_b_lep_pt",  ";Leptonic b p_{T} [GeV];Leptonic b p_{T} resolution", 50,  0., 1000., 100, -2.0, 2.0 )
histograms['reso_b_lep_y']   = TH2F( "reso_b_lep_y",   ";Leptonic b y;Leptonic b y resolution",           25, -5., 5., 100, -2.0, 2.0 )
histograms['reso_b_lep_phi'] = TH2F( "reso_b_lep_phi", ";Leptonic b #phi;Leptonic b #phi resolution",        16, -3.2, 3.2, 100, -2.0, 2.0 )
histograms['reso_b_lep_E']   = TH2F( "reso_b_lep_E",   ";Leptonic b E [GeV];Leptonic b E resolution",     50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_b_lep_m']   = TH2F( "reso_b_lep_m",   ";Leptonic b M [GeV];Leptonic b M resolution",     30, 0., 300., 100, -2.0, 2.0 )

histograms['reso_t_lep_px']  = TH2F( "reso_t_lep_px",  ";Leptonic top p_{x} [GeV];Leptonic t p_{x} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_t_lep_py']  = TH2F( "reso_t_lep_py",  ";Leptonic top p_{y} [GeV];Leptonic t p_{y} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_t_lep_pz']  = TH2F( "reso_t_lep_pz",  ";Leptonic top p_{z} [GeV];Leptonic t p_{z} resolution", 50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_t_lep_pt']  = TH2F( "reso_t_lep_pt",  ";Leptonic top p_{T} [GeV];Leptonic t p_{T} resolution", 50, 0., 500., 100, -2.0, 2.0 )
histograms['reso_t_lep_y']   = TH2F( "reso_t_lep_y",   ";Leptonic top y;Leptonic t y resolution",           25, -5., 5., 100, -2.0, 2.0 )
histograms['reso_t_lep_phi'] = TH2F( "reso_t_lep_phi", ";Leptonic top #phi;Leptonic t #phi resolution",        16, -3.2, 3.2, 100, -2.0, 2.0 )
histograms['reso_t_lep_E']   = TH2F( "reso_t_lep_E",   ";Leptonic top E [GeV];Leptonic t E resolution",     50, -1000., 1000., 100, -2.0, 2.0 )
histograms['reso_t_lep_m']   = TH2F( "reso_t_lep_m",   ";Leptonic top M [GeV];Leptonic t M resolution",     30, 0., 300., 100, -2.0, 2.0 )


for hname, h in histograms.iteritems():
  h.Sumw2()
  if hname.endswith("true")>-1:
    h.SetMarkerColor(kRed)
    h.SetLineColor(kRed)
    h.SetMarkerStyle(24)

n_events = tree.GetEntries()

print "INFO: starting event loop. Found %i events" % n_events
n_good = 0
# Print out example
for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print "INFO: Event %-9i  (%3.0f %%)" % ( i, perc )

    tree.GetEntry( i )
    
    w = tree.weight_mc
  

    W_had_true   = TLorentzVector( tree.W_had_px_true, tree.W_had_py_true, tree.W_had_pz_true, tree.W_had_E_true )
    b_had_true   = TLorentzVector( tree.b_had_px_true, tree.b_had_py_true, tree.b_had_pz_true, tree.b_had_E_true )
    t_had_true   = TLorentzVector( tree.t_had_px_true, tree.t_had_py_true, tree.t_had_pz_true, tree.t_had_E_true )
    W_lep_true   = TLorentzVector( tree.W_lep_px_true, tree.W_lep_py_true, tree.W_lep_pz_true, tree.W_lep_E_true )
    b_lep_true   = TLorentzVector( tree.b_lep_px_true, tree.b_lep_py_true, tree.b_lep_pz_true, tree.b_lep_E_true )
    t_lep_true   = TLorentzVector( tree.t_lep_px_true, tree.t_lep_py_true, tree.t_lep_pz_true, tree.t_lep_E_true )
    W_had_fitted   = TLorentzVector( tree.W_had_px_fitted, tree.W_had_py_fitted, tree.W_had_pz_fitted, tree.W_had_E_fitted )
    b_had_fitted   = TLorentzVector( tree.b_had_px_fitted, tree.b_had_py_fitted, tree.b_had_pz_fitted, tree.b_had_E_fitted )
    t_had_fitted   = TLorentzVector( tree.t_had_px_fitted, tree.t_had_py_fitted, tree.t_had_pz_fitted, tree.t_had_E_fitted )
    W_lep_fitted   = TLorentzVector( tree.W_lep_px_fitted, tree.W_lep_py_fitted, tree.W_lep_pz_fitted, tree.W_lep_E_fitted )
    b_lep_fitted   = TLorentzVector( tree.b_lep_px_fitted, tree.b_lep_py_fitted, tree.b_lep_pz_fitted, tree.b_lep_E_fitted )
    t_lep_fitted   = TLorentzVector( tree.t_lep_px_fitted, tree.t_lep_py_fitted, tree.t_lep_pz_fitted, tree.t_lep_E_fitted )

    try:
        diff_W_had_px  =  W_had_fitted.Px() - W_had_true.Px()
        diff_W_had_py  =  W_had_fitted.Py() - W_had_true.Py()
        diff_W_had_pz  =  W_had_fitted.Pz() - W_had_true.Pz()
        diff_W_had_pt  =  W_had_fitted.Pt() - W_had_true.Pt()   
        diff_W_had_y   =  W_had_fitted.Rapidity() - W_had_true.Rapidity()  
        diff_W_had_phi =  W_had_fitted.Phi()- W_had_true.Phi() 
        diff_W_had_E   =  W_had_fitted.E()  - W_had_true.E()   
        diff_W_had_m   =  W_had_fitted.M()  - W_had_true.M()

        diff_b_had_px  =  b_had_fitted.Px() - b_had_true.Px()
        diff_b_had_py  =  b_had_fitted.Py() - b_had_true.Py()
        diff_b_had_pz  =  b_had_fitted.Pz() - b_had_true.Pz()
        diff_b_had_pt  =  b_had_fitted.Pt() - b_had_true.Pt()   
        diff_b_had_y   =  b_had_fitted.Rapidity() - b_had_true.Rapidity()  
        diff_b_had_phi =  b_had_fitted.Phi() - b_had_true.Phi() 
        diff_b_had_E   =  b_had_fitted.E()  - b_had_true.E()   
        diff_b_had_m   =  b_had_fitted.M()  - b_had_true.M()

        diff_t_had_px  =  t_had_fitted.Px() - t_had_true.Px()
        diff_t_had_py  =  t_had_fitted.Py() - t_had_true.Py()
        diff_t_had_pz  =  t_had_fitted.Pz() - t_had_true.Pz()
        diff_t_had_pt  =  t_had_fitted.Pt() - t_had_true.Pt()   
        diff_t_had_y   =  t_had_fitted.Rapidity() - t_had_true.Rapidity()  
        diff_t_had_phi =  t_had_fitted.Phi() - t_had_true.Phi() 
        diff_t_had_E   =  t_had_fitted.E()   - t_had_true.E()   
        diff_t_had_m   =  t_had_fitted.M()   - t_had_true.M()
        
        reso_W_had_px  = diff_W_had_px / W_had_true.Px()
        reso_W_had_py  = diff_W_had_py / W_had_true.Py()
        reso_W_had_pz  = diff_W_had_pz / W_had_true.Pz()
        reso_W_had_pt  = diff_W_had_pt / W_had_true.Pt()   
        reso_W_had_y   = diff_W_had_y  / W_had_true.Rapidity()  
        reso_W_had_phi = diff_W_had_phi / W_had_true.Phi() 
        reso_W_had_E   = diff_W_had_E / W_had_true.E()   
        reso_W_had_m   = diff_W_had_m / W_had_true.M()

        reso_b_had_px  = diff_b_had_px / b_had_true.Px()
        reso_b_had_py  = diff_b_had_py / b_had_true.Py()
        reso_b_had_pz  = diff_b_had_pz / b_had_true.Pz()
        reso_b_had_pt  = diff_b_had_pt / b_had_true.Pt()   
        reso_b_had_y   = diff_b_had_y / b_had_true.Rapidity()  
        reso_b_had_phi = diff_b_had_phi / b_had_true.Phi() 
        reso_b_had_E   = diff_b_had_E  / b_had_true.E()   
        reso_b_had_m   = diff_b_had_m  / b_had_true.M()

        reso_t_had_px  = diff_t_had_px / t_had_true.Px()
        reso_t_had_py  = diff_t_had_py / t_had_true.Py()
        reso_t_had_pz  = diff_t_had_pz  / t_had_true.Pz()
        reso_t_had_pt  = diff_t_had_pt / t_had_true.Pt()   
        reso_t_had_y   = diff_t_had_y / t_had_true.Rapidity()  
        reso_t_had_phi = diff_t_had_phi / t_had_true.Phi() 
        reso_t_had_E   = diff_t_had_E / t_had_true.E()   
        reso_t_had_m   = diff_t_had_m  / t_had_true.M()  
    except:
        print "WARNING: invalid hadronic top, skipping event ( rn=%-10i en=%-10i )" % ( tree.runNumber, tree.eventNumber )
        PrintOut( t_had_true, t_had_fitted, "Hadronic top" )
        continue

    try:
        diff_W_lep_px  =  W_lep_fitted.Px() - W_lep_true.Px()
        diff_W_lep_py  =  W_lep_fitted.Py() - W_lep_true.Py()
        diff_W_lep_pz  =  W_lep_fitted.Pz() - W_lep_true.Pz()
        diff_W_lep_pt  =  W_lep_fitted.Pt() - W_lep_true.Pt()   
        diff_W_lep_y   =  W_lep_fitted.Rapidity() - W_lep_true.Rapidity()  
        diff_W_lep_phi =  W_lep_fitted.Phi()- W_lep_true.Phi() 
        diff_W_lep_E   =  W_lep_fitted.E()  - W_lep_true.E()   
        diff_W_lep_m   =  W_lep_fitted.M()  - W_lep_true.M()

        diff_b_lep_px  =  b_lep_fitted.Px() - b_lep_true.Px()
        diff_b_lep_py  =  b_lep_fitted.Py() - b_lep_true.Py()
        diff_b_lep_pz  =  b_lep_fitted.Pz() - b_lep_true.Pz()
        diff_b_lep_pt  =  b_lep_fitted.Pt() - b_lep_true.Pt()   
        diff_b_lep_y   =  b_lep_fitted.Rapidity() - b_lep_true.Rapidity()  
        diff_b_lep_phi =  b_lep_fitted.Phi() - b_lep_true.Phi() 
        diff_b_lep_E   =  b_lep_fitted.E()  - b_lep_true.E()   
        diff_b_lep_m   =  b_lep_fitted.M()  - b_lep_true.M()

        diff_t_lep_px  =  t_lep_fitted.Px() - t_lep_true.Px()
        diff_t_lep_py  =  t_lep_fitted.Py() - t_lep_true.Py()
        diff_t_lep_pz  =  t_lep_fitted.Pz() - t_lep_true.Pz()
        diff_t_lep_pt  =  t_lep_fitted.Pt() - t_lep_true.Pt()   
        diff_t_lep_y   =  t_lep_fitted.Rapidity() - t_lep_true.Rapidity()  
        diff_t_lep_phi =  t_lep_fitted.Phi() - t_lep_true.Phi() 
        diff_t_lep_E   =  t_lep_fitted.E()   - t_lep_true.E()   
        diff_t_lep_m   =  t_lep_fitted.M()   - t_lep_true.M()
        
        reso_W_lep_px  = diff_W_lep_px / W_lep_true.Px()
        reso_W_lep_py  = diff_W_lep_py / W_lep_true.Py()
        reso_W_lep_pz  = diff_W_lep_pz / W_lep_true.Pz()
        reso_W_lep_pt  = diff_W_lep_pt / W_lep_true.Pt()   
        reso_W_lep_y   = diff_W_lep_y  / W_lep_true.Rapidity()  
        reso_W_lep_phi = diff_W_lep_phi / W_lep_true.Phi() 
        reso_W_lep_E   = diff_W_lep_E / W_lep_true.E()   
        reso_W_lep_m   = diff_W_lep_m / W_lep_true.M()

        reso_b_lep_px  = diff_b_lep_px / b_lep_true.Px()
        reso_b_lep_py  = diff_b_lep_py / b_lep_true.Py()
        reso_b_lep_pz  = diff_b_lep_pz / b_lep_true.Pz()
        reso_b_lep_pt  = diff_b_lep_pt / b_lep_true.Pt()   
        reso_b_lep_y   = diff_b_lep_y / b_lep_true.Rapidity()  
        reso_b_lep_phi = diff_b_lep_phi / b_lep_true.Phi() 
        reso_b_lep_E   = diff_b_lep_E  / b_lep_true.E()   
        reso_b_lep_m   = diff_b_lep_m  / b_lep_true.M()

        reso_t_lep_px  = diff_t_lep_px / t_lep_true.Px()
        reso_t_lep_py  = diff_t_lep_py / t_lep_true.Py()
        reso_t_lep_pz  = diff_t_lep_pz  / t_lep_true.Pz()
        reso_t_lep_pt  = diff_t_lep_pt / t_lep_true.Pt()   
        reso_t_lep_y   = diff_t_lep_y / t_lep_true.Rapidity()  
        reso_t_lep_phi = diff_t_lep_phi / t_lep_true.Phi() 
        reso_t_lep_E   = diff_t_lep_E / t_lep_true.E()   
        reso_t_lep_m   = diff_t_lep_m  / t_lep_true.M()  
    except:
        print "WARNING: invalid leptonic top, skipping event ( rn=%-10i en=%-10i )" % ( tree.runNumber, tree.eventNumber )
        PrintOut( t_lep_true, t_lep_fitted, "Leptonic top" )
        continue

    # true
    histograms['W_had_px_true'].Fill(  W_had_true.Px(),  w )
    histograms['W_had_py_true'].Fill(  W_had_true.Py(),  w )
    histograms['W_had_pz_true'].Fill(  W_had_true.Pz(),  w )
    histograms['W_had_pt_true'].Fill(  W_had_true.Pt(),  w )
    histograms['W_had_y_true'].Fill(   W_had_true.Rapidity(), w )
    histograms['W_had_phi_true'].Fill( W_had_true.Phi(), w )
    histograms['W_had_E_true'].Fill(   W_had_true.E(),   w )
    histograms['W_had_m_true'].Fill(   W_had_true.M(),   w )

    histograms['b_had_px_true'].Fill(  b_had_true.Px(),  w )
    histograms['b_had_py_true'].Fill(  b_had_true.Py(),  w )
    histograms['b_had_pz_true'].Fill(  b_had_true.Pz(),  w )
    histograms['b_had_pt_true'].Fill(  b_had_true.Pt(),  w )
    histograms['b_had_y_true'].Fill(   b_had_true.Rapidity(), w )
    histograms['b_had_phi_true'].Fill( b_had_true.Phi(), w )
    histograms['b_had_E_true'].Fill(   b_had_true.E(),   w )
    histograms['b_had_m_true'].Fill(   b_had_true.M(),   w )

    histograms['t_had_px_true'].Fill(  t_had_true.Px(),  w )
    histograms['t_had_py_true'].Fill(  t_had_true.Py(),  w )
    histograms['t_had_pz_true'].Fill(  t_had_true.Pz(),  w )
    histograms['t_had_pt_true'].Fill(  t_had_true.Pt(),  w )
    histograms['t_had_y_true'].Fill(   t_had_true.Rapidity(), w )
    histograms['t_had_phi_true'].Fill( t_had_true.Phi(), w )
    histograms['t_had_E_true'].Fill(   t_had_true.E(),   w )
    histograms['t_had_m_true'].Fill(   t_had_true.M(),   w )

    histograms['W_lep_px_true'].Fill(  W_lep_true.Px(),  w )
    histograms['W_lep_py_true'].Fill(  W_lep_true.Py(),  w )
    histograms['W_lep_pz_true'].Fill(  W_lep_true.Pz(),  w )
    histograms['W_lep_pt_true'].Fill(  W_lep_true.Pt(),  w )
    histograms['W_lep_y_true'].Fill(   W_lep_true.Rapidity(), w )
    histograms['W_lep_phi_true'].Fill( W_lep_true.Phi(), w )
    histograms['W_lep_E_true'].Fill(   W_lep_true.E(),   w )
    histograms['W_lep_m_true'].Fill(   W_lep_true.M(),   w )

    histograms['b_lep_px_true'].Fill(  b_lep_true.Px(),  w )
    histograms['b_lep_py_true'].Fill(  b_lep_true.Py(),  w )
    histograms['b_lep_pz_true'].Fill(  b_lep_true.Pz(),  w )
    histograms['b_lep_pt_true'].Fill(  b_lep_true.Pt(),  w )
    histograms['b_lep_y_true'].Fill(   b_lep_true.Rapidity(), w )
    histograms['b_lep_phi_true'].Fill( b_lep_true.Phi(), w )
    histograms['b_lep_E_true'].Fill(   b_lep_true.E(),   w )
    histograms['b_lep_m_true'].Fill(   b_lep_true.M(),   w )

    histograms['t_lep_px_true'].Fill(  t_lep_true.Px(),  w )
    histograms['t_lep_py_true'].Fill(  t_lep_true.Py(),  w )
    histograms['t_lep_pz_true'].Fill(  t_lep_true.Pz(),  w )
    histograms['t_lep_pt_true'].Fill(  t_lep_true.Pt(),  w )
    histograms['t_lep_y_true'].Fill(   t_lep_true.Rapidity(), w )
    histograms['t_lep_phi_true'].Fill( t_lep_true.Phi(), w )
    histograms['t_lep_E_true'].Fill(   t_lep_true.E(),   w )
    histograms['t_lep_m_true'].Fill(   t_lep_true.M(),   w )

    # Fitted
    histograms['W_had_px_fitted'].Fill(  W_had_fitted.Px(),  w )
    histograms['W_had_py_fitted'].Fill(  W_had_fitted.Py(),  w )
    histograms['W_had_pz_fitted'].Fill(  W_had_fitted.Pz(),  w )
    histograms['W_had_pt_fitted'].Fill(  W_had_fitted.Pt(),  w )
    histograms['W_had_y_fitted'].Fill(   W_had_fitted.Rapidity(), w )
    histograms['W_had_phi_fitted'].Fill( W_had_fitted.Phi(), w )
    histograms['W_had_E_fitted'].Fill(   W_had_fitted.E(),   w )
    histograms['W_had_m_fitted'].Fill(   W_had_fitted.M(),   w )

    histograms['b_had_px_fitted'].Fill(  b_had_fitted.Px(),  w )
    histograms['b_had_py_fitted'].Fill(  b_had_fitted.Py(),  w )
    histograms['b_had_pz_fitted'].Fill(  b_had_fitted.Pz(),  w )
    histograms['b_had_pt_fitted'].Fill(  b_had_fitted.Pt(),  w )
    histograms['b_had_y_fitted'].Fill(   b_had_fitted.Rapidity(), w )
    histograms['b_had_phi_fitted'].Fill( b_had_fitted.Phi(), w )
    histograms['b_had_E_fitted'].Fill(   b_had_fitted.E(),   w )
    histograms['b_had_m_fitted'].Fill(   b_had_fitted.M(),   w )

    histograms['t_had_px_fitted'].Fill(  t_had_fitted.Px(),  w )
    histograms['t_had_py_fitted'].Fill(  t_had_fitted.Py(),  w )
    histograms['t_had_pz_fitted'].Fill(  t_had_fitted.Pz(),  w )
    histograms['t_had_pt_fitted'].Fill(  t_had_fitted.Pt(),  w )
    histograms['t_had_y_fitted'].Fill(   t_had_fitted.Rapidity(), w )
    histograms['t_had_phi_fitted'].Fill( t_had_fitted.Phi(), w )
    histograms['t_had_E_fitted'].Fill(   t_had_fitted.E(),   w )
    histograms['t_had_m_fitted'].Fill(   t_had_fitted.M(),   w )

    histograms['W_lep_px_fitted'].Fill(  W_lep_fitted.Px(),  w )
    histograms['W_lep_py_fitted'].Fill(  W_lep_fitted.Py(),  w )
    histograms['W_lep_pz_fitted'].Fill(  W_lep_fitted.Pz(),  w )
    histograms['W_lep_pt_fitted'].Fill(  W_lep_fitted.Pt(),  w )
    histograms['W_lep_y_fitted'].Fill(   W_lep_fitted.Rapidity(), w )
    histograms['W_lep_phi_fitted'].Fill( W_lep_fitted.Phi(), w )
    histograms['W_lep_E_fitted'].Fill(   W_lep_fitted.E(),   w )
    histograms['W_lep_m_fitted'].Fill(   W_lep_fitted.M(),   w )

    histograms['b_lep_px_fitted'].Fill(  b_lep_fitted.Px(),  w )
    histograms['b_lep_py_fitted'].Fill(  b_lep_fitted.Py(),  w )
    histograms['b_lep_pz_fitted'].Fill(  b_lep_fitted.Pz(),  w )
    histograms['b_lep_pt_fitted'].Fill(  b_lep_fitted.Pt(),  w )
    histograms['b_lep_y_fitted'].Fill(   b_lep_fitted.Rapidity(), w )
    histograms['b_lep_phi_fitted'].Fill( b_lep_fitted.Phi(), w )
    histograms['b_lep_E_fitted'].Fill(   b_lep_fitted.E(),   w )
    histograms['b_lep_m_fitted'].Fill(   b_lep_fitted.M(),   w )

    histograms['t_lep_px_fitted'].Fill(  t_lep_fitted.Px(),  w )
    histograms['t_lep_py_fitted'].Fill(  t_lep_fitted.Py(),  w )
    histograms['t_lep_pz_fitted'].Fill(  t_lep_fitted.Pz(),  w )
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

    # residuals
    histograms['diff_W_had_px'].Fill(  diff_W_had_px,  w )
    histograms['diff_W_had_py'].Fill(  diff_W_had_py,  w )
    histograms['diff_W_had_pz'].Fill(  diff_W_had_pz,  w )
    histograms['diff_W_had_pt'].Fill(  diff_W_had_pt,  w )
    histograms['diff_W_had_y'].Fill(   diff_W_had_y,   w )
    histograms['diff_W_had_phi'].Fill( diff_W_had_phi, w )
    histograms['diff_W_had_E'].Fill(   diff_W_had_E,   w )
    histograms['diff_W_had_m'].Fill(   diff_W_had_m,   w )

    histograms['diff_b_had_px'].Fill(  diff_b_had_px,  w )
    histograms['diff_b_had_py'].Fill(  diff_b_had_py,  w )
    histograms['diff_b_had_pz'].Fill(  diff_b_had_pz,  w )
    histograms['diff_b_had_pt'].Fill(  diff_b_had_pt,  w )
    histograms['diff_b_had_y'].Fill(   diff_b_had_y,   w )
    histograms['diff_b_had_phi'].Fill( diff_b_had_phi, w )
    histograms['diff_b_had_E'].Fill(   diff_b_had_E,   w )
    histograms['diff_b_had_m'].Fill(   diff_b_had_m,   w )

    histograms['diff_t_had_px'].Fill(  diff_t_had_px,  w )
    histograms['diff_t_had_py'].Fill(  diff_t_had_py,  w )
    histograms['diff_t_had_pz'].Fill(  diff_t_had_pz,  w )
    histograms['diff_t_had_pt'].Fill(  diff_t_had_pt,  w )
    histograms['diff_t_had_y'].Fill(   diff_t_had_y,   w )
    histograms['diff_t_had_phi'].Fill( diff_t_had_phi, w )
    histograms['diff_t_had_E'].Fill(   diff_t_had_E,   w )
    histograms['diff_t_had_m'].Fill(   diff_t_had_m,   w )
    
    histograms['diff_W_lep_px'].Fill(  diff_W_lep_px,  w )
    histograms['diff_W_lep_py'].Fill(  diff_W_lep_py,  w )
    histograms['diff_W_lep_pz'].Fill(  diff_W_lep_pz,  w )
    histograms['diff_W_lep_pt'].Fill(  diff_W_lep_pt,  w )
    histograms['diff_W_lep_y'].Fill(   diff_W_lep_y, w )
    histograms['diff_W_lep_phi'].Fill( diff_W_lep_phi, w )
    histograms['diff_W_lep_E'].Fill(   diff_W_lep_E,   w )
    histograms['diff_W_lep_m'].Fill(   diff_W_lep_m,   w )

    histograms['diff_b_lep_px'].Fill(  diff_b_lep_px,  w )
    histograms['diff_b_lep_py'].Fill(  diff_b_lep_py,  w )
    histograms['diff_b_lep_pz'].Fill(  diff_b_lep_pz,  w )
    histograms['diff_b_lep_pt'].Fill(  diff_b_lep_pt,  w )
    histograms['diff_b_lep_y'].Fill(   diff_b_lep_y, w )
    histograms['diff_b_lep_phi'].Fill( diff_b_lep_phi, w )
    histograms['diff_b_lep_E'].Fill(   diff_b_lep_E,   w )
    histograms['diff_b_lep_m'].Fill(   diff_b_lep_m,   w )

    histograms['diff_t_lep_px'].Fill(  diff_t_lep_px,  w )
    histograms['diff_t_lep_py'].Fill(  diff_t_lep_py,  w )
    histograms['diff_t_lep_pz'].Fill(  diff_t_lep_pz,  w )
    histograms['diff_t_lep_pt'].Fill(  diff_t_lep_pt,  w )
    histograms['diff_t_lep_y'].Fill(   diff_t_lep_y, w )
    histograms['diff_t_lep_phi'].Fill( diff_t_lep_phi, w )
    histograms['diff_t_lep_E'].Fill(   diff_t_lep_E,   w )
    histograms['diff_t_lep_m'].Fill(   diff_t_lep_m,   w )
    
    # resolution
    histograms['reso_W_had_px'].Fill(  W_had_true.Px(), reso_W_had_px,  w )
    histograms['reso_W_had_py'].Fill(  W_had_true.Py(), reso_W_had_py,  w )
    histograms['reso_W_had_pz'].Fill(  W_had_true.Pz(), reso_W_had_pz,  w )
    histograms['reso_W_had_pt'].Fill(  W_had_true.Pt(), reso_W_had_pt,  w )
    histograms['reso_W_had_y'].Fill(   W_had_true.Rapidity(),  reso_W_had_y,   w )
    histograms['reso_W_had_phi'].Fill( W_had_true.Phi(), reso_W_had_phi, w )
    histograms['reso_W_had_E'].Fill(   W_had_true.E(),  reso_W_had_E,   w )
    histograms['reso_W_had_m'].Fill(   W_had_true.M(),  reso_W_had_m,   w )

    histograms['reso_b_had_px'].Fill(  b_had_true.Px(), reso_b_had_px,  w )
    histograms['reso_b_had_py'].Fill(  b_had_true.Py(), reso_b_had_py,  w )
    histograms['reso_b_had_pz'].Fill(  b_had_true.Pz(), reso_b_had_pz,  w )
    histograms['reso_b_had_pt'].Fill(  b_had_true.Pt(), reso_b_had_pt,  w )
    histograms['reso_b_had_y'].Fill(   b_had_true.Rapidity(),  reso_b_had_y,   w )
    histograms['reso_b_had_phi'].Fill( b_had_true.Phi(), reso_b_had_phi, w )
    histograms['reso_b_had_E'].Fill(   b_had_true.E(),  reso_b_had_E,   w )
    histograms['reso_b_had_m'].Fill(   b_had_true.M(),  reso_b_had_m,   w )

    histograms['reso_t_had_px'].Fill(  t_had_true.Px(), reso_t_had_px,  w )
    histograms['reso_t_had_py'].Fill(  t_had_true.Py(), reso_t_had_py,  w )
    histograms['reso_t_had_pz'].Fill(  t_had_true.Pz(), reso_t_had_pz,  w )
    histograms['reso_t_had_pt'].Fill(  t_had_true.Pt(), reso_t_had_pt,  w )
    histograms['reso_t_had_y'].Fill(   t_had_true.Rapidity(),  reso_t_had_y,   w )
    histograms['reso_t_had_phi'].Fill( t_had_true.Phi(), reso_t_had_phi, w )
    histograms['reso_t_had_E'].Fill(   t_had_true.E(),  reso_t_had_E,   w )
    histograms['reso_t_had_m'].Fill(   t_had_true.M(),  reso_t_had_m,   w )

    histograms['reso_W_lep_px'].Fill(  W_lep_true.Px(), reso_W_lep_px,  w )
    histograms['reso_W_lep_py'].Fill(  W_lep_true.Py(), reso_W_lep_py,  w )
    histograms['reso_W_lep_pz'].Fill(  W_lep_true.Pz(), reso_W_lep_pz,  w )
    histograms['reso_W_lep_pt'].Fill(  W_lep_true.Pt(), reso_W_lep_pt,  w )
    histograms['reso_W_lep_y'].Fill(   W_lep_true.Rapidity(),  reso_W_lep_y,   w )
    histograms['reso_W_lep_phi'].Fill( W_lep_true.Phi(), reso_W_lep_phi, w )
    histograms['reso_W_lep_E'].Fill(   W_lep_true.E(),  reso_W_lep_E,   w )
    histograms['reso_W_lep_m'].Fill(   W_lep_true.M(),  reso_W_lep_m,   w )

    histograms['reso_b_lep_px'].Fill(  b_lep_true.Px(), reso_b_lep_px,  w )
    histograms['reso_b_lep_py'].Fill(  b_lep_true.Py(), reso_b_lep_py,  w )
    histograms['reso_b_lep_pz'].Fill(  b_lep_true.Pz(), reso_b_lep_pz,  w )
    histograms['reso_b_lep_pt'].Fill(  b_lep_true.Pt(), reso_b_lep_pt,  w )
    histograms['reso_b_lep_y'].Fill(   b_lep_true.Rapidity(),  reso_b_lep_y,   w )
    histograms['reso_b_lep_phi'].Fill( b_lep_true.Phi(), reso_b_lep_phi, w )
    histograms['reso_b_lep_E'].Fill(   b_lep_true.E(),  reso_b_lep_E,   w )
    histograms['reso_b_lep_m'].Fill(   b_lep_true.M(),  reso_b_lep_m,   w )

    histograms['reso_t_lep_px'].Fill(  t_lep_true.Px(), reso_t_lep_px,  w )
    histograms['reso_t_lep_py'].Fill(  t_lep_true.Py(), reso_t_lep_py,  w )
    histograms['reso_t_lep_pz'].Fill(  t_lep_true.Pz(), reso_t_lep_pz,  w )
    histograms['reso_t_lep_pt'].Fill(  t_lep_true.Pt(), reso_t_lep_pt,  w )
    histograms['reso_t_lep_y'].Fill(   t_lep_true.Rapidity(),  reso_t_lep_y,   w )
    histograms['reso_t_lep_phi'].Fill( t_lep_true.Phi(), reso_t_lep_phi, w )
    histograms['reso_t_lep_E'].Fill(   t_lep_true.E(),  reso_t_lep_E,   w )
    histograms['reso_t_lep_m'].Fill(   t_lep_true.M(),  reso_t_lep_m,   w )

    n_good += 1
    
    if i < 10:
      PrintOut( t_had_true, t_had_fitted, "Hadronic top" )
      PrintOut( t_lep_true, t_lep_fitted, "Leptonic top" )
      
ofile.Write()
ofile.Close()

print "Finished. Saved output file:", ofilename

f_good = 100. * float( n_good ) / float( n_events )
print "Good events: %.2f" % f_good
