#!/usr/bin/python3
# Copyright 2008 The RE2 Authors.  All Rights Reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Generate C++ tables for Unicode Script and Category groups."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unicode

_header = """
// GENERATED BY make_unicode_groups.py; DO NOT EDIT.
// make_unicode_groups.py >unicode_groups.cc

#include "re2/unicode_groups.h"

namespace re2 {

"""

_trailer = """

}  // namespace re2

"""

n16 = 0
n32 = 0

def MakeRanges(codes):
  """Turn a list like [1,2,3,7,8,9] into a range list [[1,3], [7,9]]"""
  ranges = []
  last = -100
  for c in codes:
    if c == last+1:
      ranges[-1][1] = c
    else:
      ranges.append([c, c])
    last = c
  return ranges

def PrintRanges(type, name, ranges):
  """Print the ranges as an array of type named name."""
  print("static const %s %s[] = {" % (type, name))
  for lo, hi in ranges:
    print("\t{ %d, %d }," % (lo, hi))
  print("};")

# def PrintCodes(type, name, codes):
#   """Print the codes as an array of type named name."""
#   print("static %s %s[] = {" % (type, name))
#   for c in codes:
#     print("\t%d," % (c,))
#   print("};")

def PrintGroup(name, codes):
  """Print the data structures for the group of codes.
  Return a UGroup literal for the group."""

  # See unicode_groups.h for a description of the data structure.

  # Split codes into 16-bit ranges and 32-bit ranges.
  range16 = MakeRanges([c for c in codes if c < 65536])
  range32 = MakeRanges([c for c in codes if c >= 65536])

  # Pull singleton ranges out of range16.
  # code16 = [lo for lo, hi in range16 if lo == hi]
  # range16 = [[lo, hi] for lo, hi in range16 if lo != hi]

  global n16
  global n32
  n16 += len(range16)
  n32 += len(range32)

  ugroup = "{ \"%s\", +1" % (name,)
  # if len(code16) > 0:
  #   PrintCodes("uint16_t", name+"_code16", code16)
  #   ugroup += ", %s_code16, %d" % (name, len(code16))
  # else:
  #   ugroup += ", 0, 0"
  if len(range16) > 0:
    PrintRanges("URange16", name+"_range16", range16)
    ugroup += ", %s_range16, %d" % (name, len(range16))
  else:
    ugroup += ", 0, 0"
  if len(range32) > 0:
    PrintRanges("URange32", name+"_range32", range32)
    ugroup += ", %s_range32, %d" % (name, len(range32))
  else:
    ugroup += ", 0, 0"
  ugroup += " }"
  return ugroup

def main():
  categories = unicode.Categories()
  scripts = unicode.Scripts()
  print(_header)
  ugroups = []
  for name in sorted(categories):
    ugroups.append(PrintGroup(name, categories[name]))
  for name in sorted(scripts):
    ugroups.append(PrintGroup(name, scripts[name]))
  print("// %d 16-bit ranges, %d 32-bit ranges" % (n16, n32))
  print("const UGroup unicode_groups[] = {")
  ugroups.sort()
  for ug in ugroups:
    print("\t%s," % (ug,))
  print("};")
  print("const int num_unicode_groups = %d;" % (len(ugroups),))
  print(_trailer)

if __name__ == '__main__':
  main()
