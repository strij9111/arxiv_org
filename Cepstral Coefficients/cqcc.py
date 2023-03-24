from python_speech_features import logfbank
import scipy.io.wavfile as wav

rate, sig = wav.read("path/to/your/audiofile.wav")
cqcc_feat = logfbank(sig, rate)
