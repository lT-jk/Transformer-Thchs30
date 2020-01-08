from utils.audio import melspectrogram,inv_mel_spectrogram,load_wav,save_wav


wav_path = "LJ001-0008.wav"
raw_wav = load_wav(wav_path)
mel_spec = melspectrogram(raw_wav)
inv_wav = inv_mel_spectrogram(mel_spec)
save_wav(inv_wav,"inv.wav")

