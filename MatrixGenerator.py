#!/usr/bin/python
#coding=utf-8
import sys
import argparse
import time
import random

progress_step = 5

parser = argparse.ArgumentParser()
parser.add_argument('size', help='matrix size', type=int)
args = parser.parse_args()

n = args.size


if len(sys.argv) != 2:
	print 'Wrong argument. Usage {0}.py size'.format(sys.argv[0])

f_matrix = open('matrix.txt', 'w')
f_vector = open('vector.txt', 'w')

f_vector.write('{0}\r\n'.format(str(n)))
f_matrix.write('{0}\r\n'.format(str(n)))

maximum = n * n

line = [0] * n

print 'Start generation'
prev_progress = 0
current_progress = 0
t0 = time.clock()
for i in xrange(n):
	for j in xrange(n):
		if j == i:
			line[j] = maximum
		else:
			line[j] = int(random.uniform(0, n))
	f_matrix.write(' '.join(str(x) for x in line))

	if i != n - 1:
		f_matrix.write('\r\n')
	f_vector.write('{0} '.format(str(sum(line))))

	current_progress = int(i * 100.0 / n)
	if current_progress >= prev_progress + progress_step :
		print '{0}%'.format(current_progress)
		prev_progress = current_progress


print 'Generation completed! In: {0} second'.format(time.clock() - t0)