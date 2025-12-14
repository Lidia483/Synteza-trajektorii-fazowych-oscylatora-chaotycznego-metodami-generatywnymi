#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""\
raw2csv.py: Open LTSpice RAW file, resample it and save as CSV.
"""

__author__      = "Łukasz Makowski"
__copyright__   = "Copyright 2025"
__credits__ = ["spicelib contributors"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Łukasz Makowski"
__email__ = "lukasz.makowski.ee@pw.edu.pl"
__status__ = "Production"

import sys
import numpy as np
from scipy import interpolate
from spicelib import RawRead
import matplotlib.pyplot as plt


def showdata(xnew, ChL, ChR):
  plt.plot(xnew[:1000], ChL[:1000], label = 'Channel Left', \
      c='#a77c', ls='-', lw=6, \
      marker='*', mec='#0004', mfc='#a77', ms=8)
  plt.plot(xnew[:1000], ChR[:1000],  label = 'Channel Right', \
      c='#77ac', ls='-', lw=6, \
      marker='o', mec='#0004', mfc='#77a', ms=6)

  plt.legend()
  plt.show()


def norm11(a):
    ratio = 2 / (np.max(a) - np.min(a))
    shift = (np.max(a) + np.min(a)) / 2
    return (a - shift) * ratio


def processfile(rawfile):
  #print(rawfile.get_trace_names())
  #print(rawfile.get_raw_property())

  steps = rawfile.get_steps() # if there is more steps, otherwise just 0

  # TIME AXIS
  xtrace = rawfile.get_trace('time')  # Gets the time axis
  xorg = xtrace.get_wave(0)
  timestep = 2.5E-6 
  xnew = np.arange(min(xorg), max(xorg), timestep)

  # VALUES AXIS: Channel Left
  ChL = rawfile.get_trace("V(v1)")
  ChL = ChL.get_wave(0)
  ChL = norm11(ChL)
  ChLinterpol = interpolate.interp1d(xorg, ChL)
  ChL = ChLinterpol(xnew)

  # VALUES AXIS: Channel Right
  ChR = rawfile.get_trace("V(v2)")
  ChR = ChR.get_wave(0)
  ChR = norm11(ChR)
  ChRinterpol = interpolate.interp1d(xorg, ChR)
  ChR = ChRinterpol(xnew)

  outarray = np.stack([xnew, ChL, ChR], axis = 1)
  np.savetxt("foo.csv", outarray, delimiter=",")

  showdata(xnew, ChL, ChR)


def main(argv):
  print("Loading...")
  rawfile = RawRead(argv[1])
  print("Done!")
  processfile(rawfile)


def usage(argv):
  print("Usage:\n\t{0} filename".format(argv[0]))


if __name__ == "__main__":
  try:
      sys.argv[1]
  except IndexError:
      usage(sys.argv)
      sys.exit(1)
  else:
      main(sys.argv)

