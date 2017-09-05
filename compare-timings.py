#!/usr/bin/env python
#
# A utility program for comparing outputs of timing programs (time-*.cpp).
# Useful to testing whether a new version of a kernel is better than the old version.

import re
import sys


if len(sys.argv) != 3:
    print >>sys.stderr, 'usage: compare-timings.py <old_file> <new_file>'
    sys.exit(1)


def parse_file(filename):
    """Returns pair of lists(kernel_name_list, ns_per_sample_list)"""

    kret = [ ]
    tret = [ ]
    
    for line in open(filename):
        if (len(line) > 0) and line[-1] == '\n':
            line = line[:-1]
            
        m = re.match(r'(.*): (\d+) iterations, total time (\d+\.\d*) sec \((\d+\.\d*) sec/iteration, (\d+\.\d*) ns/sample', line)
        if m:
            kret.append(m.group(1))
            tret.append(float(m.group(5)))
            continue
        
        if re.search(r'nthreads=(\d+), nfreq=(\d+), nt_chunk=(\d+), stride=(\d+)', line):
            continue
        
        print >>sys.stderr, "%s: warning: couldn't parse line: %s" % (filename, line)

    if len(kret) == 0:
        print >>sys.stderr, '%s: parse failure (or empty file)' % filename
        sys.exit(1)
    
    return (kret, tret)


(kernel_name1, ns_per_sample1) = parse_file(sys.argv[1])
(kernel_name2, ns_per_sample2) = parse_file(sys.argv[2])

if kernel_name1 != kernel_name2:
    print sys.stderr, 'compare-timings.py: fatal: both files must have precisely the same kernel names'
    sys.exit(1)

print 'Note: ratios are (%s timing) / (%s timing)' % (sys.argv[1], sys.argv[2])

for (k,t1,t2) in zip(kernel_name1, ns_per_sample1, ns_per_sample2):
    print '%s: ratio=%g' % (k, (t1/t2))

