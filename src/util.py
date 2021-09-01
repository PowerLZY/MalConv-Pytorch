# coding=utf-8
from torch.utils.data import Dataset
import pandas as pd
import glob as glob
import struct
import string
import seaborn
import numpy as np
import matplotlib.pyplot as plt

seaborn.set()
COFF_LENGTH = 24
SECT_ENTRY_LEN = 40

def write_pred(test_pred,test_idx,file_path):
	test_pred = [item for sublist in test_pred for item in sublist]
	with open(file_path,'w') as f:
		for idx,pred in zip(test_idx,test_pred):
			print(idx.upper()+','+str(pred[0]),file=f)

# Dataset preparation
class ExeDataset(Dataset):
	def __init__(self, fp_list, data_path, label_list, first_n_byte=2000000):
		self.fp_list = fp_list
		self.data_path = data_path
		self.label_list = label_list
		self.first_n_byte = first_n_byte

	def __len__(self):
		return len(self.fp_list)

	def __getitem__(self, idx):
		try:
			with open(self.data_path + self.fp_list[idx],'rb') as f:
				tmp = [i+1 for i in f.read()[:self.first_n_byte]] # index 0 will be special padding index 每个值加一
				tmp = tmp+[0]*(self.first_n_byte-len(tmp))
		except:
			with open(self.data_path + self.fp_list[idx].lower(),'rb') as f:
				tmp = [i+1 for i in f.read()[:self.first_n_byte]]
				tmp = tmp+[0]*(self.first_n_byte-len(tmp))

		return np.array(tmp), np.array([self.label_list[idx]])

# black.csv 合并 white.csv
def merge(black, white, file_name, media_directory):
	train_raw_data = black.append(white)
	#  for _ 中表示缺省
	train_raw_data["labels"] = [1 for _ in range(black.shape[0])] + [0 for _ in range(white.shape[0])]
	train_raw_data.to_csv(f"{media_directory}/{file_name}.csv", index=False)#, header=False)

	return train_raw_data

def get_data_label(data_path, label_path, file_name):
	names = []
	df = pd.DataFrame()
	for path in glob.glob(data_path):
		names.append(path.split('/')[-1])

	df["id"] = names
	df.to_csv(label_path + file_name, index=False, encoding="utf-8")
# 返回目录下文件列表
def get_filename(datapath):
	filename = []

	for path in glob.glob(datapath):
		filename.append(path.split('/')[-1])

	return filename

def plot_header_contribution_histogram(bytestring_program: bytearray, itgs: np.ndarray, percentage: bool = True,
									   force_plot: bool = True, save_path: str = " ", filename = "test"):
	"""Plot integrated gradient results, divided by sections

	Parameters
	----------
	bytestring_program: bytearray
		the program as bytearray
	itgs : numpy array
		array containing the result of Integrated gradient
	percentage :bool
		display percentage instead of absolute values (default: True)
	force_plot : bool
		Should show the results? (default:True)
	"""

	pe_position = struct.unpack("<I", bytestring_program[0x3C: 0x3C + 4])[0]
	n_sects = struct.unpack(
		"<I", bytestring_program[pe_position + 6: pe_position + 8] + b"\x00\x00"
	)[0]
	opt_h_len = struct.unpack(
		"<I", bytestring_program[pe_position + 20: pe_position + 22] + b"\x00\x00"
	)[0]
	sect_offset = opt_h_len + pe_position + COFF_LENGTH
	sect_end = sect_offset + SECT_ENTRY_LEN * n_sects
	mean_dos = np.mean(np.array(itgs[0:pe_position]))
	mean_header_coff = np.sum(np.array(itgs[pe_position: pe_position + 24]))
	mean_header_optional = np.sum(
		np.array(itgs[pe_position + COFF_LENGTH: pe_position + COFF_LENGTH + opt_h_len])
	)
	mean_section_table = np.sum(np.array(itgs[sect_offset:sect_end]))
	names = ["DOS Header", "COFF Header", "Optional Header", "Section Headers"]
	to_plot = [mean_dos, mean_header_coff, mean_header_optional, mean_section_table]
	sect_name_length = 8
	for i in range(n_sects):
		offset_index = sect_offset + i * SECT_ENTRY_LEN + sect_name_length + 12
		size_index = sect_offset + i * SECT_ENTRY_LEN + sect_name_length + 8
		offset = struct.unpack("<I", bytestring_program[offset_index: offset_index + 4])[0]
		size = struct.unpack("<I", bytestring_program[size_index: size_index + 4])[0]
		name = str(
			bytestring_program[
			sect_offset + i * SECT_ENTRY_LEN: sect_offset + i * SECT_ENTRY_LEN + 8
			]
				.decode("utf-8")
				.rstrip("\x00")
		)
		mean = np.sum(np.array(itgs[offset: offset + size]))
		to_plot.append(mean)
		names.append(name)
	if percentage:
		to_plot = to_plot / np.linalg.norm(to_plot, ord=2)
	fig = plt.figure()
	x = range(len(names))
	positives = [i if i >= 0 else 0 for i in to_plot]
	negatives = [i if i < 0 else 0 for i in to_plot]
	plt.bar(x, positives, width=0.2, color="r")
	plt.bar(x, negatives, width=0.2, color="b")
	plt.yticks(fontsize=22)
	plt.xticks(x, names, fontsize=22, rotation=45, ha="right")
	xs = np.linspace(-1, 7, 4)
	plt.plot(xs, np.zeros(len(xs)), "k")
	ax = plt.gca()
	ax.set_facecolor((1.0, 1.0, 1.0))
	plt.title(
		"Sum of each contribution,divided into headers and sections\n", fontsize=25
	)
	if force_plot:
		fig.tight_layout()
		fig.set_size_inches(18.5, 10.5)
		fig.savefig(save_path + "/plot_header_contribution_histogram_"+filename+".png")


def plot_code_segment(
		pe_file: list,
		start: int,
		stop: int,
		itgs: np.ndarray,
		title: str,
		show_positives: bool = True,
		show_negatives: bool = False,
		force_plot: bool = True,
		width: int = 16,
		percentage: bool = True,
		save_path: str = " ",
		filename = "test"
):
	"""Plot contribution of chunks of bytes.

	Parameters
	----------
	pe_file : list
		list of bytes
	start : int
		starting index for segment to plot
	stop : int
		stop index for segment to plot
	itgs : numpy array
		array containing result of integrated gradients
	title : str
		plot title
	show_positives : bool
		show positives contributes (default:True)
	show_negatives : bool
		show negative contributes (default:False}
	force_plot : bool
		show plot (default:True)
	width : int
		how many byte per row of the heatmap (default:16)
	percentage : bool
		display percentage instead of absolute values (default: True)
	"""
	grad_section = itgs[start:stop]
	text = [
		hex(i-1) if chr(i-1) not in string.ascii_letters else chr(i-1)
		for i in pe_file[start:stop]
	]
	cols = width
	tot_len = len(text)
	row = tot_len // cols
	if tot_len % cols:
		rem = tot_len % cols
		text.extend(["" for _ in range(cols - rem)])
		grad_section.extend([0 for _ in range(cols - rem)])
		row = row + 1
	text = np.array(text)
	grad_section = np.array(grad_section) / (
		np.linalg.norm(grad_section) if percentage else 1
	)

	if not show_positives:
		grad_section[grad_section > 0] = 0
	if not show_negatives:
		grad_section[grad_section < 0] = 0

	grad_section = grad_section.reshape((row, cols))
	text = text.reshape((row, cols))
	fig = plt.figure()
	hmap = seaborn.heatmap(
		grad_section,
		annot=text,
		fmt="",
		cmap="seismic",
		center=0,
		annot_kws={"size": 18},
	)
	hmap.collections[0].colorbar.ax.tick_params(labelsize=20)
	hmap.set_title(title, fontsize=25)
	fig.tight_layout()
	fig.set_size_inches(18.5, 10.5)
	fig.savefig(save_path+"/plot_code_segment_"+filename+".png")




