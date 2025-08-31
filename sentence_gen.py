#!/usr/bin/env python3
import os, sys, re, shutil, subprocess, random, math

# --- CONFIG ------------------------------------------------------------------
SOURCE_DIR       = "words"
OUTPUT_DIR       = "wavs"
SENTENCES_FILE   = "sentences.txt"
CSV_FILE         = "metadata.csv"

TARGET_SR        = 22050
TARGET_CODEC     = "pcm_s16le"   # 16-bit PCM (2 bytes per sample)
CHANNELS         = 1

# Start vs end trim (protect fricative onsets like /f/)
START_SILENCE_THRESH_DB = -47.0
START_MIN_SIL_DUR_S     = 0.030
END_SILENCE_THRESH_DB   = -40.0
END_MIN_SIL_DUR_S       = 0.030

# Words shorter than this are considered unusable and will be rebuilt
MIN_WORD_DUR_S   = 0.020   # 20 ms safety floor

# Micro fades per word
FADE_IN_S         = 0.002   # keep fricative onsets crisp
FADE_OUT_S        = 0.010

# Gentle cleanup
HIGHPASS_HZ       = 45

# Per-word loudness control
NORMALIZE_TARGET_DBFS = -20.0
PEAK_LIMIT_LINEAR     = 0.89
GAIN_MIN_DB           = -12.0
GAIN_MAX_DB           = +12.0

# Crossfading
OVERLAP_S         = 0.080
CROSSFADE_C1      = "qsin"
CROSSFADE_C2      = "qsin"
JITTER_S          = 0.030   # ±30 ms per join

# Final sentence tail (reverse-fade to guarantee end-only)
SENTENCE_TAIL_FADE_S = 0.030

INTER_WORD_GAP_S  = 0.00
TRIM_CACHE_DIR    = ".trimmed_cache"

PRINT_GAIN_STATS  = True
PRINT_FADE_DEBUG  = False
# -----------------------------------------------------------------------------


def run_ffmpeg(args):
    try:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("\n[ffmpeg cmd]", " ".join(args))
        subprocess.run(args, check=False)
        sys.exit(1)

def ffprobe_duration_sec(path):
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            text=True
        ).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0

# ---- cache validation --------------------------------------------------------
WAV_HEADER_BYTES = 44
BYTES_PER_SAMPLE = 2  # pcm_s16le
def min_bytes_for_duration(dur_s):
    samples = int(math.ceil(dur_s * TARGET_SR * CHANNELS))
    return WAV_HEADER_BYTES + samples * BYTES_PER_SAMPLE

def cache_too_short(path):
    """Return True if cached file is obviously bad and must be rebuilt."""
    if not os.path.exists(path):
        return True
    try:
        size_ok = os.path.getsize(path) >= min_bytes_for_duration(MIN_WORD_DUR_S * 0.9)
    except OSError:
        size_ok = False
    dur = ffprobe_duration_sec(path)
    dur_ok = dur >= (MIN_WORD_DUR_S * 0.95)  # small tolerance for probe rounding
    return not (size_ok and dur_ok)
# -----------------------------------------------------------------------------

def build_wav_map(src_dir):
    m = {}
    for fname in os.listdir(src_dir):
        if fname.lower().endswith(".wav"):
            m[os.path.splitext(fname)[0].lower()] = os.path.join(src_dir, fname)
    return m

def _trim_and_fade_filter():
    """Per-word cleanup + trims with min-silence, plus micro fades."""
    parts = ["pan=mono|c0=c0"]
    if HIGHPASS_HZ and HIGHPASS_HZ > 0:
        parts.append(f"highpass=f={HIGHPASS_HZ}")
    # START trim (gentle)
    parts.append(
        f"silenceremove=start_periods=1:"
        f"start_duration={START_MIN_SIL_DUR_S}:"
        f"start_threshold={START_SILENCE_THRESH_DB}dB"
    )
    # END trim via reverse; also micro fade on the (reversed) start
    parts += [
        "areverse",
        f"silenceremove=start_periods=1:start_duration={END_MIN_SIL_DUR_S}:"
        f"start_threshold={END_SILENCE_THRESH_DB}dB",
        f"afade=t=in:d={FADE_OUT_S}",
        "areverse",
        f"afade=t=in:d={FADE_IN_S}",
    ]
    return ",".join(parts)

_VOL_RE = re.compile(r"mean_volume:\s*(-?(?:\d+(?:\.\d+)?|inf|-inf|nan))\s*dB", re.I)

def measure_mean_dbfs_with_filter(src_path, base_filter):
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-v", "info",
        "-i", src_path, "-ar", str(TARGET_SR),
        "-af", f"{base_filter},volumedetect",
        "-f", "null", "-"
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = (p.stderr or "") + (p.stdout or "")
    m = _VOL_RE.search(text)
    if not m:
        return None
    val = m.group(1).lower()
    if val in ("inf", "-inf", "nan"):
        return None
    return float(val)

def trim_normalize_to_cache(src, dst):
    # base = trims (start/end) + micro fades + highpass
    base = _trim_and_fade_filter()

    # 1) Normal path (measure -> gain -> limit)
    measured = measure_mean_dbfs_with_filter(src, base)
    gain_db = 0.0 if measured is None else (NORMALIZE_TARGET_DBFS - measured)
    gain_db = max(GAIN_MIN_DB, min(GAIN_MAX_DB, gain_db))

    if PRINT_GAIN_STATS:
        name = os.path.basename(src)
        mv = f"{measured:.1f}" if measured is not None else "NA"
        print(f"[norm] {name:>24}  mean={mv:>5} dB  gain={gain_db:+.2f} dB")

    final_filter = f"{base},volume={gain_db}dB,alimiter=limit={PEAK_LIMIT_LINEAR}"
    run_ffmpeg([
        "ffmpeg","-y","-i", src,
        "-ar", str(TARGET_SR),
        "-af", final_filter,
        "-ac", str(CHANNELS),
        "-c:a", TARGET_CODEC,
        dst
    ])

    # 2) If too-short, rebuild with no silenceremove (fallback), then pad if needed
    if cache_too_short(dst):
        fallback_base_parts = ["pan=mono|c0=c0"]
        if HIGHPASS_HZ and HIGHPASS_HZ > 0:
            fallback_base_parts.append(f"highpass=f={HIGHPASS_HZ}")
        fallback_base_parts += [
            "areverse", f"afade=t=in:d={FADE_OUT_S}", "areverse", f"afade=t=in:d={FADE_IN_S}"
        ]
        fallback_base = ",".join(fallback_base_parts)

        measured2 = measure_mean_dbfs_with_filter(src, fallback_base)
        gain2 = 0.0 if measured2 is None else (NORMALIZE_TARGET_DBFS - measured2)
        gain2 = max(GAIN_MIN_DB, min(GAIN_MAX_DB, gain2))

        print(f"[warn] {os.path.basename(src)} cache too short; rebuilding without silenceremove (gain {gain2:+.2f} dB).")

        final_fallback = f"{fallback_base},volume={gain2}dB,alimiter=limit={PEAK_LIMIT_LINEAR}"
        run_ffmpeg([
            "ffmpeg","-y","-i", src,
            "-ar", str(TARGET_SR),
            "-af", final_fallback,
            "-ac", str(CHANNELS),
            "-c:a", TARGET_CODEC,
            dst
        ])

        # If still extremely short, pad a hair so acrossfade won't choke
        if cache_too_short(dst):
            need_pad_s = max(0.0, MIN_WORD_DUR_S - ffprobe_duration_sec(dst) + 0.005)
            if need_pad_s > 0:
                run_ffmpeg([
                    "ffmpeg","-y","-i", dst,
                    "-af", f"apad=pad_dur={need_pad_s}",
                    "-c:a", TARGET_CODEC, dst
                ])

def acrossfade_words(word_paths, out_wav, overlap_requested):
    """Acrossfade chain with per-pair jitter; final reverse-fade to avoid chop."""
    if len(word_paths) == 1:
        run_ffmpeg([
            "ffmpeg","-y","-i", word_paths[0],
            "-af", f"areverse,afade=t=in:d={SENTENCE_TAIL_FADE_S},areverse",
            "-ar", str(TARGET_SR), "-ac", str(CHANNELS), "-c:a", TARGET_CODEC, out_wav
        ])
        return

    durations = [ffprobe_duration_sec(p) for p in word_paths]
    shortest_global = min([d for d in durations if d > 0.0], default=overlap_requested or 0.02)
    d_base = max(0.005, min(overlap_requested, shortest_global * 0.45)) if overlap_requested > 0 else 0.0

    # Build xfade filter chain
    inputs = []
    for p in word_paths: inputs += ["-i", p]
    steps = []
    for i in range(len(word_paths) - 1):
        left  = "[0:a]" if i == 0 else f"[a{i:02d}]"
        right = f"[{i+1}:a]"
        out   = f"[a{i+1:02d}]"
        d_pair_short = min(durations[i] or shortest_global, durations[i+1] or shortest_global)
        d_cap = max(0.005, min(overlap_requested, d_pair_short * 0.45))
        d_pair = max(0.005, min(d_cap, d_base + random.uniform(-JITTER_S, JITTER_S)))
        if PRINT_FADE_DEBUG:
            print(f"[xfade] {i}-{i+1}: d={d_pair:.3f}s (cap {d_cap:.3f})")
        steps.append(f"{left}{right}acrossfade=d={d_pair}:c1={CROSSFADE_C1}:c2={CROSSFADE_C2}{out}")

    # Final tail fade using reverse trick
    last_label = f"[a{len(word_paths)-1:02d}]"
    steps.append(f"{last_label}areverse,afade=t=in:d={SENTENCE_TAIL_FADE_S},areverse[aout]")
    filt = ";".join(steps)

    run_ffmpeg([
        "ffmpeg","-y", *inputs,
        "-filter_complex", filt, "-map", "[aout]",
        "-ar", str(TARGET_SR), "-ac", str(CHANNELS), "-c:a", TARGET_CODEC, out_wav
    ])

# ------------------------------- main ----------------------------------------
wav_map = build_wav_map(SOURCE_DIR)

try:
    with open(SENTENCES_FILE, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
except FileNotFoundError:
    sys.stderr.write(f"Missing {SENTENCES_FILE}\n"); sys.exit(1)

valid_entries, errors = [], []
for idx, sentence in enumerate(lines, start=1):
    if not sentence.strip(): continue
    words = sentence.lower().split()
    missing = [w for w in words if w not in wav_map]
    (errors.append((idx, sentence, missing)) if missing else valid_entries.append((idx, sentence)))

if errors:
    sys.stderr.write(f"Error: missing word WAVs in {len(errors)} line(s):\n")
    for line_no, sentence, missing in errors:
        sys.stderr.write(f"  Line {line_no}: “{sentence}” – missing: {', '.join(missing)}\n")
    sys.exit(1)

os.makedirs(TRIM_CACHE_DIR, exist_ok=True)
needed_words = sorted({w for _, s in valid_entries for w in s.lower().split()})
trimmed_paths = {}
for w in needed_words:
    src = wav_map[w]
    dst = os.path.join(TRIM_CACHE_DIR, os.path.basename(src))
    trimmed_paths[w] = dst
    # Rebuild cache if missing, older than source, or INVALID (size/duration)
    if (not os.path.exists(dst)
        or os.path.getmtime(dst) < os.path.getmtime(src)
        or cache_too_short(dst)):
        trim_normalize_to_cache(src, dst)

# Build sentences
gap_wav = None
if OVERLAP_S <= 0 and INTER_WORD_GAP_S > 0:
    gap_wav = os.path.join(TRIM_CACHE_DIR, f"_gap_{int(INTER_WORD_GAP_S*1000)}ms.wav")
    if not os.path.exists(gap_wav):
        run_ffmpeg(["ffmpeg","-y","-f","lavfi","-t",str(INTER_WORD_GAP_S),
                    "-i", f"anullsrc=r={TARGET_SR}:cl=mono","-c:a",TARGET_CODEC,gap_wav])

os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_lines = []
for out_idx, (line_no, sentence) in enumerate(valid_entries, start=1):
    out_wav = os.path.join(OUTPUT_DIR, f"vox_{out_idx:04d}.wav")
    paths = [trimmed_paths[w] for w in sentence.lower().split()]
    if OVERLAP_S > 0:
        acrossfade_words(paths, out_wav, OVERLAP_S)
        # belt & braces: if somehow empty, fallback to concat + tail fade
        if ffprobe_duration_sec(out_wav) < 0.01:
            print(f"[warn] {os.path.basename(out_wav)} empty; using concat fallback.")
            list_txt = f"tmp_list_fallback_{out_idx:04d}.txt"
            with open(list_txt, "w", encoding="utf-8") as lf:
                for p in paths: lf.write(f"file '{os.path.abspath(p)}'\n")
            run_ffmpeg([
                "ffmpeg","-y","-f","concat","-safe","0","-i", list_txt,
                "-af", f"areverse,afade=t=in:d={SENTENCE_TAIL_FADE_S},areverse",
                "-ar", str(TARGET_SR), "-ac", str(CHANNELS), "-c:a", TARGET_CODEC, out_wav
            ])
            os.remove(list_txt)
    else:
        list_txt = f"tmp_list_{out_idx:04d}.txt"
        with open(list_txt,"w",encoding="utf-8") as lf:
            for i,p in enumerate(paths):
                lf.write(f"file '{os.path.abspath(p)}'\n")
                if gap_wav and i != len(paths)-1:
                    lf.write(f"file '{os.path.abspath(gap_wav)}'\n")
        run_ffmpeg([
            "ffmpeg","-y","-f","concat","-safe","0","-i",list_txt,
            "-af", f"areverse,afade=t=in:d={SENTENCE_TAIL_FADE_S},areverse",
            "-ar", str(TARGET_SR), "-ac", str(CHANNELS), "-c:a", TARGET_CODEC, out_wav
        ])
        os.remove(list_txt)

    csv_lines.append(f"{OUTPUT_DIR}/vox_{out_idx:04d}.wav|{sentence}")

with open(CSV_FILE,"w",encoding="utf-8") as out:
    out.write("\n".join(csv_lines))

print(f"✔ Generated {len(valid_entries)} sentence WAVs "
      f"({'overlap ' + str(int(OVERLAP_S*1000)) + 'ms' if OVERLAP_S>0 else 'no overlap'}) "
      f"and wrote '{CSV_FILE}'.")
