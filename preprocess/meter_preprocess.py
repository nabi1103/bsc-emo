import sys
import os
import csv
import re
import xml.etree.ElementTree as ET

path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(1, path)

from reader import Reader

class MeterPreprocess:
	def __init__(self):
		self.dummy = None

	def get_versification(self, meter_line):
		label = None
		meter = ''.join(meter_line)
		meter = re.sub('\+', 'I', meter)
		meter = re.sub('\-', 'o', meter)
		#print(meter)
		iambicseptaplus = re.compile("oIoIoIoIoIoIoIo?")
		hexameter =       re.compile('Ioo?Ioo?Ioo?Ioo?IooIo$')
		alxiambichexa =   re.compile("oIoIoIoIoIoIo?$")
		iambicpenta =     re.compile("oIoIoIoIoIo?$")
		iambictetra =     re.compile("oIoIoIoIo?$")
		iambictri =       re.compile("oIoIoIo?$")
		iambicdi =        re.compile("oIoIo?$")
		iambic =          re.compile("oIoIo?")
		trochseptaplus =  re.compile('IoIoIoIoIoIoIo?')
		trochhexa =       re.compile('IoIoIoIoIoIo?$')
		trochpenta =      re.compile('IoIoIoIoIo?$')
		trochtetra =      re.compile('IoIoIoIo?$')
		trochtri =        re.compile('IoIoIo?$')
		trochdi =         re.compile('IoIo?$')
		troch =           re.compile('IoIo?')
		artemajor =       re.compile('oIooIooIooIo$')
		artemajorhalf =   re.compile('oIooIo$')
		zehnsilber =      re.compile('...I.....I$') 
		amphidi =         re.compile('oIooIo')
		amphitri =        re.compile('oIooIooIo')
		adontrochamphi =  re.compile('IooIo$')
		adoneusspond =    re.compile('IooII$')
		iambamphi =       re.compile('oIooI$')
		iambchol =        re.compile('IooI')
		anapaestdiplus =  re.compile('ooIooI')
		daktyldiplus =    re.compile('IooIoo')
		anapaestinit =    re.compile('ooI')
		daktylinit =      re.compile('Ioo')
		#alexandriner =    re.compile('oIoIoIoIoIoIo?$')
		#adoneus =        re.compile('IooIo$')

		verses = {'iambic.septa.plus':iambicseptaplus,\
			'hexameter':hexameter,\
			'alexandr.iambic.hexa':alxiambichexa,\
			'iambic.penta':iambicpenta,\
			'iambic.tetra':iambictetra,\
			'iambic.tri':iambictri,\
			'iambic.di':iambicdi,\
			'troch.septa.plus':trochseptaplus,\
			'troch.hexa':trochhexa,\
			'troch.penta':trochpenta,\
			'troch.tetra':trochtetra,\
			'troch.tri':trochtri,\
			'troch.di':trochdi,\
			'arte_major':artemajor,\
			'arte_major.half':artemajorhalf,\
			'zehnsilber':zehnsilber,\
			'adoneus.troch.amphi':adontrochamphi,\
			'adoneus.spond':adoneusspond,\
			'amphi.tri.plus':amphitri,\
			'amphi.iamb':iambamphi,\
			'amphi.di.plus':amphidi,\
			'chol.iamb':iambchol,\
			'iambic.mix':iambic,\
			'troch.mix':troch,\
			'anapaest.di.plus':anapaestdiplus,\
			'daktyl.di.plus':daktyldiplus,\
			'anapaest.init':anapaestinit,\
			'daktyl.init':daktylinit}

		for label, pattern in verses.items():
			result = pattern.match(meter)
			if label == 'chol.iamb':
				result = pattern.search(meter)
			if result != None:
				return label
		else: return 'other'

	def read_poem(self, path):
		try:
			root = ET.parse(path).getroot()
		except:
			print(path)
			return []
		poem = []

		for lg in root.iter('{http://www.tei-c.org/ns/1.0}lg'):
			# Look for stanza
			if (lg.get('sample') == 'complete' or lg.get('met') != None):
				sroot = lg
				# Read every line
				for l in sroot.iter('{http://www.tei-c.org/ns/1.0}l'):
					meter = self.get_versification(l.get('met'))
					line = ''
					for s in l.iter('{http://www.tei-c.org/ns/1.0}seg'):
						sub = ''
						for t in s.itertext():
							sub += t
						line += sub
					poem.append([line.lower().strip(), meter])

		if poem == []:
			for lg in root.iter('lg'):
				# Look for stanza
				if (lg.get('met') != None):
					sroot = lg
					# Read every line
					for l in sroot.iter('l'):
						meter = l.get('met')
						if meter == None:
							meter = l.get('real')
						meter = self.get_versification(meter)
						line = ''
						for s in l.iter('seg'):
							sub = ''
							for t in s.itertext():
								sub += t
							line += sub
						poem.append([line.lower().strip(), meter])
		return poem

	def label_count(self, data, label):
		tags = [d[1] for d in data]
		count = 0
		for t in tags:
			if t == label:
				count += 1 
		return count

	def remove_minor_label(self, data):
		label = list(set([d[1] for d in data]))
		keep = []

		for l in label:
			count = self.label_count(data, l)
			if count > 10:
				has_label = [d for d in data if d[1] == l and d[0] != '']
				keep += has_label
			else:
				has_label = [[d[0], 'other'] for d in data if d[1] == l and d[0] != '']
				keep += has_label
		return keep

	def assign_label_meter(self, line):
		labels = ['iambic.di', 'iambic.tri', 'iambic.tetra', 'chol.iamb', 'alexandr.iambic.hexa', 'troch.tetra', 'iambic.penta', 'other']
		vector = [0, 0, 0, 0, 0, 0, 0, 0]
		lst_label = [l.strip() for l in line[1].split(':')]
		for l in lst_label:
			idx = labels.index(l)
			vector[idx] = 1
		return [line[0], vector]


# r = Reader()
# folder_0 = r.get_all_in_dir('F://[Uni]//Thesis//[Misc]//Code//for_better_for_verse-master//poems', '.xml')
# folder_1 = r.get_all_in_dir('F://[Uni]//Thesis//[Misc]//Code//for_better_for_verse-master//poems2', '.xml')
# folder = folder_0 + folder_1

# mp = MeterPreprocess()

# data = []
# poem = [mp.read_poem(f) for f in folder]

# for p in poem[10:15]:
# 	for l in p:
# 		print(l)
# 	print('')

# for p in poem:
#     data += p

# data = mp.remove_minor_label(data)

# label = ['iambic.di', 'iambic.tri', 'iambic.tetra', 'chol.iamb', 'alexandr.iambic.hexa', 'troch.tetra', 'iambic.penta', 'other']

# for l in label:
#     print(l, mp.label_count(data, l))

# print(len(data))

# print('------')

# # for d in data:
# # 	print(mp.assign_label_meter(d))

# from random import shuffle
# shuffle(data)
# train = data[:800]
# test = data[800:]

# for l in label:
#     print(l, mp.label_count(train, l))

# print('-----')

# for l in label:
#     print(l, mp.label_count(test, l))

# with open('F://[Uni]//Thesis//[Misc]//Code//for_better_for_verse-master//meter-train.txt', 'w', encoding='utf-8', newline='') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     tsv_writer.writerows(train)
    
# with open('F://[Uni]//Thesis//[Misc]//Code//for_better_for_verse-master//meter-test.txt', 'w', encoding='utf-8', newline='') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     tsv_writer.writerows(test)
