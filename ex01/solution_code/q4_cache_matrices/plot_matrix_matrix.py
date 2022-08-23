#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys

path = 'matrix_matrix_times.txt'
argv = sys.argv
if len(argv) > 1:
    path = argv[1]
data = np.loadtxt(path, skiprows=1)
data = np.atleast_2d(data)

with open(path) as f:
    header = f.readline()
header = header.split()

def select(block=None, N=None, colmaj=False, unopt=False):
    global data
    global header
    sel_block = block
    sel_N = N
    sel_colmaj = colmaj
    sel_unopt = unopt
    vblock = []
    vN = []
    vtime = []
    assert header[0] == 'N'
    for col in range(len(header)):
        for row in range(len(data)):
            h = header[col]
            if h == "N":
                continue
            elif h == "unopt":
                unopt = True
                block = None
            else:
                unopt = False
                block = int(re.findall(".._(\d*)", h)[0])
                assert h[:2] in ["bc", "br"], "unknown header: %r" % h
                colmaj = (h[1] == 'c')
            N = int(data[row][0] + 0.5)
            if sel_block is not None and block != sel_block:
                continue
            if sel_N is not None and N != sel_N:
                continue
            if colmaj != sel_colmaj:
                continue
            if unopt != sel_unopt:
                continue
            vblock.append(block)
            vN.append(N)
            vtime.append(data[row][col])
    return np.array(vblock), np.array(vN), np.array(vtime)


all_block, all_N, _ = select()
all_block = sorted(list(set(all_block)))
all_N = sorted(list(set(all_N)))

def size_to_kilobytes(block):
    block = np.array(block)
    return block**2 * 8 / (1 << 10)

# multiplications per second
def time_to_ops(t, N):
    return N ** 3 / t / 1e9

plt.figure(figsize=(10, 6))
showkilo = False
N2color = dict()

for i, N in enumerate(all_N):
    _, _, unopt_t = select(N=N, unopt=True)
    unopt_mul = time_to_ops(unopt_t, N)
    line = plt.axhline(unopt_mul, c='C{:}'.format(i % 6))
    N2color[N] = line.get_color()

for N in all_N:
    vb, vN, vt = select(N=N)
    vops = time_to_ops(vt, N)
    vbshow = size_to_kilobytes(vb) if showkilo else vb
    line, = plt.plot(vbshow,
                     vops,
                     '--o',
                     markersize=8,
                     c=N2color[N],
                     label="N={:}".format(N))

for N in all_N:
    vb, vN, vt = select(N=N, colmaj=True)
    vops = time_to_ops(vt, N)
    vbshow = size_to_kilobytes(vb) if showkilo else vb
    line, = plt.plot(vbshow,
                     vops,
                     '-s',
                     markersize=8,
                     c=N2color[N],
                     label="N={:}, colmaj".format(N))

plt.ylim(0, None)
plt.xscale('log')
ax = plt.gca()
ax.set_xticks(size_to_kilobytes(all_block) if showkilo else all_block)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(axis='x',
               which='minor',
               bottom=False,
               top=False,
               labelbottom=False)

plt.ylabel('multiplications per second [1e9]')
plt.xlabel('block buffer size [KiB]' if showkilo else 'block size')

secax = ax.secondary_xaxis(
    'top',
    functions=(
        lambda b: b**2 * 8 / 1024,
        lambda s: (s / 8)**0.5))
secax.tick_params(axis='x',
                  which='minor',
                  bottom=False,
                  top=False,
                  labelbottom=False)
secax.set_xlabel('block buffer size [KiB]')
secax.set_xticks(size_to_kilobytes(all_block))
secax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: "{:.1f}".format(x)))

plt.legend(loc=0, handlelength=4)
plt.grid(True)

plt.savefig('matrix-matrix.pdf')
