# coding=utf-8
#输出结果不想带情绪标识，修改def format_str函数部分


import os
import librosa
import base64
import io
import gradio as gr
import re
import sys
from io import StringIO
import numpy as np
import torch
import torchaudio


from funasr import AutoModel

model = "iic/SenseVoiceSmall"
model = AutoModel(model=model,
				  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
				  vad_kwargs={"max_single_segment_time": 30000},
				  trust_remote_code=True,
				  device="cuda:0"
				  )

import re


def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")



emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict = {
	"<|zh|>": "<|lang|>",
	"<|en|>": "<|lang|>",
	"<|yue|>": "<|lang|>",
	"<|ja|>": "<|lang|>",
	"<|ko|>": "<|lang|>",
	"<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷", }


def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s

def format_str_v1(s):
		# 正则表达式模式.纯文本返回
	pattern = r'<\|[^|]+\|>'

	# 替换匹配的内容为空字符串
	result = re.sub(pattern, '', s)

	return result.strip()

def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()


def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None

	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		# else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def format_str_v4(s):
	#返回格式实例：  [happy]打算打算打算的****
	pattern = re.compile(r'<\|.*?\|>')
	matches = pattern.findall(s)
	# 提取尖括号内的文本内容并存储到新列表
	content_list = [match[2:-2] for match in matches]
	emotion=content_list[1] if content_list else 'none'
	if not emotion =='EMO_UNKNOWN' :
		emotion=emotion.lower()
		tex=f'[{emotion}]'+format_str_v1(s)
		return tex
	else:
		return '[normal]'+format_str_v1(s)

def model_inference(input_wav, language, output_type,fs=16000):
	global Result
	typ_dic={"纯文本":'v1', "事件+文本+情感": 'v2', "文本+情感emo":'v3',"[情感]+文本":'v4'}
	# task_abbr = {"Speech Recognition": "ASR", "Rich Text Transcription": ("ASR", "AED", "SER")}
	language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
					 "nospeech": "nospeech"}

	# task = "Speech Recognition" if task is None else task
	language = "auto" if len(language) < 1 else language
	selected_language = language_abbr[language]
	# selected_task = task_abbr.get(task)

	# print(f"input_wav: {type(input_wav)}, {input_wav[1].shape}, {input_wav}")

	if isinstance(input_wav, tuple):
		fs, input_wav = input_wav
		input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
		if len(input_wav.shape) > 1:
			input_wav = input_wav.mean(-1)
		if fs != 16000:
			print(f"audio_fs: {fs}")
			resampler = torchaudio.transforms.Resample(fs, 16000)
			input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
			input_wav = resampler(input_wav_t[None, :])[0, :].numpy()

	merge_vad = True  # False if selected_task == "ASR" else True
	# print(f"language: {language}, merge_vad: {merge_vad}")
	try:
		text = model.generate(input=input_wav,
							  cache={},
							  language=language,
							  use_itn=True,
							  batch_size_s=60)
	except :
		print(input_wav, '识别失败，跳过')
		return False
	# merge_vad=merge_vad)

	print(input_wav,'翻译结果：',text)
	text = text[0]["text"]
	if not text:
		print(input_wav, '翻译结果为空，跳过')
		return False
	_typ=typ_dic[output_type]
	format_func_name=f'format_str_{_typ}'
	print(format_func_name)
	if format_func_name in globals():
		format_func = globals()[format_func_name]
		text = format_func(text)
		print('----输出文本已过滤----')
		print(text)

	# print(text)
	Result.append(text)
	return text


def recognize_and_rename(audio_file,result,rename,savetxt):
	global Message
	try:
		# 获取识别结果的前200个字符作为新文件名
		file_name=os.path.basename(audio_file)
		long_text =format_str_v1(result)
		file_extension = os.path.splitext(audio_file)[-1].lower()
		if len(long_text) > 200:
			new_filename = long_text[:200].replace('/', '_').replace('\\', '_') + file_extension
		else:
			new_filename = long_text.replace('/', '_').replace('\\', '_') + file_extension

		# 重命名音频文件
		new_audio_file = os.path.join(os.path.dirname(audio_file), new_filename)
		if rename == True:
			os.rename(audio_file, new_audio_file)
			Message.append(f"{file_name} 重命名为---- {new_filename}")
		if savetxt == True and rename == True:
			# 将识别结果写入文件
			with open(f'{new_audio_file}.txt', 'w+', encoding='utf-8') as f:
				f.write(long_text + '\n')
			Message.append(f"{file_name} 识别结果已保存到文件---- {new_filename}.txt")
		if savetxt == True and rename == False:
			with open(f'{audio_file}.txt', 'w+', encoding='utf-8') as f:
				f.write(long_text + '\n')
				Message.append(f"{file_name} 识别结果已保存到文件---- {file_name}.txt")

	except Exception as e:
		print(f"重命名失败,跳过: {audio_file}")
		
		print(e)
		return False


def get_audio_files(folder_path, check_subfolders=False):
	if not os.path.exists(folder_path):
		print(f"文件夹 {folder_path} 不存在")
		return '找不到输入路径'

	# 定义音频文件的扩展名列表
	audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
	audio_files = []
	pattern = r'\[.*?\]'
	if check_subfolders == True:
		# 使用 os.walk 递归遍历文件夹及其子文件夹

		for root, dirs, files in os.walk(folder_path):
			for filename in files:
				#检测是否已转录了
				if not bool(re.search(pattern, filename)):
					# 构建文件的完整路径
					file_path = os.path.join(root, filename)

					# 提取文件扩展名
					file_extension = os.path.splitext(filename)[-1].lower()

					# 检查是否为文件且文件扩展名在音频文件扩展名列表中
					if os.path.isfile(file_path) and file_extension in audio_extensions:
						audio_files.append(file_path)
	else:
		for files in os.listdir(folder_path):
			#检测是否已转录了
			if not bool(re.search(pattern, files)):
				# 构建文件的完整路径
				file_path = os.path.join(folder_path, files)

				# 提取文件扩展名
				file_extension = os.path.splitext(files)[-1].lower()

				# 检查是否为文件且文件扩展名在音频文件扩展名列表中
				if os.path.isfile(file_path) and file_extension in audio_extensions:
					audio_files.append(file_path)
	return audio_files

def main(folder_path, rename_audio, save_text, check_subfolders, language, output_type, progress=gr.Progress()):
    global Message, Result
    Message = ['------------保存情况--------------']
    Result = ['------------配置情况--------------']
    audie_files = get_audio_files(folder_path, check_subfolders)
    Result.extend([
        f"文件夹路径: {folder_path}",
        f"是否递归搜索子文件夹: {check_subfolders}",
        f"发现的音频文件数量: {len(audie_files)}",
        f"设置识别语音: {language}",
        f"是否重命名音频文件: {rename_audio}",
        f"是否保存识别结果: {save_text}",
        f"保存模式: {output_type}",
        '----------识别结果---------'
    ])
    num_files = len(audie_files)
    for i, audio_file in enumerate(audie_files):
        progress((i + 1) / num_files, desc=f"处理文件 {i + 1}/{num_files}: {audio_file}")
        print(f"开始处理文件: {audio_file}")
        result = model_inference(audio_file, language, output_type)
        if not result:
            continue
        # print('保存模式：', output_type)
        if rename_audio or save_text:
            recognize_and_rename(audio_file, result, rename_audio, save_text)
    print("\n".join(Message))
    Message.append('--------任务全部完成------')
    Result.extend(Message)
    return "\n".join(Result)

audio_examples = [
	["example/zh.mp3", "zh"],
	["example/yue.mp3", "yue"],
	["example/en.mp3", "en"],
	["example/ja.mp3", "ja"],
	["example/ko.mp3", "ko"],
	["example/emo_1.wav", "auto"],
	["example/emo_2.wav", "auto"],
	["example/emo_3.wav", "auto"],
	# ["example/emo_4.wav", "auto"],
	# ["example/event_1.wav", "auto"],
	# ["example/event_2.wav", "auto"],
	# ["example/event_3.wav", "auto"],
	["example/rich_1.wav", "auto"],
	["example/rich_2.wav", "auto"],
	# ["example/rich_3.wav", "auto"],
	["example/longwav_1.wav", "auto"],
	["example/longwav_2.wav", "auto"],
	["example/longwav_3.wav", "auto"],
	# ["example/longwav_4.wav", "auto"],
]

html_content = """
<div>
	<h2 style="font-size: 22px;margin-left: 0px;">Voice Understanding Model: SenseVoice-Small</h2>
	<p style="font-size: 18px;margin-left: 20px;">SenseVOT-Small 是一种仅编码器的语音基础模型，专为快速语音理解而设计。它包含多种功能，包括语音识别（ASR）、口语识别（LID）、语音情感识别（SER）和音频事件检测（AED）。SenseVOT-Small 支持中文、英文、粤语、日语和韩语的多语言识别。此外，它还提供极低的推理延迟，比 Whisper-Small 快 7 倍，比 Whisper - large快 17 倍。</p>
	<h2 style="font-size: 22px;margin-left: 0px;">Usage</h2> <p style="font-size: 18px;margin-left: 20px;">上传一个音频文件或通过麦克风输入，然后选择任务和语言。音频会被转录为相应的文本，同时附带相关情绪（😊开心、😡生气 / 激动、😔难过）以及各类声音事件类型（😀笑声、🎼音乐、👏掌声、🤧咳嗽和打喷嚏、😭哭泣）。事件标签会放在文本前面，情绪则放在文本后面。.</p>
	<p style="font-size: 18px;margin-left: 20px;">推荐的音频输入持续时间低于 30 秒。对于超过 30 秒的音频，建议进行本地部署。</p>
	<h2 style="font-size: 22px;margin-left: 0px;">Repo</h2>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice</a>: 多语言语音理解模型</p>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/modelscope/FunASR" target="_blank">FunASR</a>: 基本语音识别工具包</p>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">CosyVoice</a>: 高质量的多语言 TTS 模型</p>
</div>
"""

html_content_1 = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;"><br><br>添加批量音频文件识别+文件重命名+识别结果保存功能（功能本人添加，非原项目代码，酌情使用）（licor）</h2>
</div>
"""
def file_number(file_path, check_subfolders):
	filedslist=get_audio_files(file_path,check_subfolders)
	return len(filedslist)


Message=[]
Result=[]


def launch():
	with gr.Blocks(theme=gr.themes.Soft()) as demo:
		# gr.Markdown(description)
		gr.HTML(html_content)
		with gr.Row():
			with gr.Column():
				audio_inputs = gr.Audio(label="单个转写上传音频或使用麦克风")

				with gr.Accordion("配置"):
					language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
												  value="auto",
												  label="语言选择")
				fn_button = gr.Button("开始转写", variant="primary")
				text_outputs = gr.Textbox(label="转写结果")
			gr.Examples(examples=audio_examples, inputs=[audio_inputs, language_inputs], examples_per_page=20)

		fn_button.click(model_inference, inputs=[audio_inputs, language_inputs], outputs=text_outputs)

		gr.HTML(html_content_1)

		with gr.Row():
			with gr.Column():
				folder_path_input = gr.Textbox(label="文件夹路径批量处理", placeholder="请输入文件夹路径，两端不要带引号")
				with gr.Row():
					rename_audio = gr.Checkbox(label="重命名音频文件")
					save_text = gr.Checkbox(label="保存识别文本")
					check_subfolders = gr.Checkbox(label="检查子文件夹")


		with gr.Row():
			with gr.Accordion("配置"):
				language_inputs_1 = gr.Dropdown(choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
											  value="auto",
											  label="选择音频语言")
				ouputt_type = gr.Dropdown(choices=["纯文本", "事件+文本+情感", "文本+情感emo","[情感]+文本"],
											  value="文本+情感emo",
											  label="选择输出保存模式")
			with gr.Column():
				with gr.Row():
					file_nubmer = gr.Number(label="找到音频文件数量", interactive=False)
					fresh_button = gr.Button("刷新路径", variant="primary")
					fresh_button.click(file_number, inputs=[folder_path_input, check_subfolders], outputs=file_nubmer)
				start_button = gr.Button("开始批量转录", variant="primary")

		with gr.Row():
			progress_text = gr.Textbox(label="进度", interactive=False)

			start_button.click(main, inputs=[folder_path_input, rename_audio, save_text, check_subfolders,
											 language_inputs_1,ouputt_type], outputs=progress_text)

	demo.launch()




if __name__ == "__main__":
	# iface.launch()
	check_cuda_availability()
	launch()
