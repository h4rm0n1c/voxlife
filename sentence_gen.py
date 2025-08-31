#!/usr/bin/env python3
import os, sys, re, shutil, subprocess, random, math, hashlib

# --- CONFIG ------------------------------------------------------------------
SOURCE_DIR       = "words"
OUTPUT_DIR       = "wavs"
SENTENCES_FILE   = "test_sentences.txt"
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

# Sentence-level crossfading (set to 0.00 for concat w/ gaps only)
OVERLAP_S         = 0.008
CROSSFADE_C1      = "qsin"
CROSSFADE_C2      = "qsin"
JITTER_S          = 0.030   # ±30 ms per join (crossfade mode only)

# Ligature (within-word) crossfades like "think" + "ing" => "thinking"
LIGATURE_SUFFIXES     = {"ing"}     # extendable later if needed
LIGATURE_OVERLAP_S    = 0.120       # tighter join than normal; no jitter
LIGATURE_CACHE_PREFIX = "_lig_"     # names for cached fused WAVs

# Final sentence tail (reverse-fade to guarantee end-only)
SENTENCE_TAIL_FADE_S = 0.030

INTER_WORD_GAP_S  = 0.00  # ignored if using random gaps below
TRIM_CACHE_DIR    = ".trimmed_cache"

PRINT_GAIN_STATS  = True
PRINT_FADE_DEBUG  = False
# -----------------------------------------------------------------------------

# --- NATURALIZED TIMING PARAMS -----------------------------------------------
PACING_JITTER = 0.129  # Up to ±45ms jitter per join (non-crossfade only)
SENTENCE_PACE_RANGE = (0.88, 1.16)  # Each sentence is 92–108% speed of baseline
SEMANTIC_PAUSE_WORDS = {
    "and", "please", "now", "then", "exit", "after", "before",
    "immediately", "attention", "wait", "system"
}
NUMBER_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",
    "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety","hundred","thousand"
}
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

def fuse_two_with_acrossfade(left_path, right_path, out_path, overlap_d):
    """Create a tight within-word join of two trimmed clips (no jitter)."""
    # Rebuild if missing or stale
    try:
        need = (not os.path.exists(out_path) or
                os.path.getmtime(out_path) < os.path.getmtime(left_path) or
                os.path.getmtime(out_path) < os.path.getmtime(right_path) or
                cache_too_short(out_path))
    except OSError:
        need = True
    if not need:
        return

    run_ffmpeg([
        "ffmpeg","-y","-i", left_path, "-i", right_path,
        "-filter_complex",
        f"[0:a][1:a]acrossfade=d={max(0.005, overlap_d)}:c1={CROSSFADE_C1}:c2={CROSSFADE_C2}[a]",
        "-map","[a]", "-ar", str(TARGET_SR), "-ac", str(CHANNELS), "-c:a", TARGET_CODEC, out_path
    ])

def acrossfade_words(word_paths, out_wav, overlap_requested, pace_mult=1.0):
    """Acrossfade chain with per-pair jitter + optional global pace, then tail fade."""
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

    # Build xfade filter chain, now with pace_mult applied and added jitter
    inputs = []
    for p in word_paths: inputs += ["-i", p]
    steps = []
    for i in range(len(word_paths) - 1):
        left  = "[0:a]" if i == 0 else f"[a{i:02d}]"
        right = f"[{i+1}:a]"
        out   = f"[a{i+1:02d}]"
        d_pair_short = min(durations[i] or shortest_global, durations[i+1] or shortest_global)
        d_cap = max(0.005, min(overlap_requested, d_pair_short * 0.45))
        # Jitter + pace
        d_jitter = random.uniform(-JITTER_S, JITTER_S)
        d_pair = max(0.005, min(d_cap, (d_base + d_jitter) * pace_mult))
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

# -------------------------- ligature/transcription ----------------------------

# Only remap these tokens in the transcript (not in audio):
SPELL_LETTERS_IN_TRANSCRIPT = {"usa", "gib"}

def spell_letters(token_lower: str) -> str:
    """Return 'U S A' for 'usa', 'G I B' for 'gib', else the token unchanged."""
    if token_lower in SPELL_LETTERS_IN_TRANSCRIPT:
        return " ".join(list(token_lower.upper()))
    return token_lower

def sanitize_for_filename(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", x)

def plan_ligatures_and_transcript(words, wav_map):
    """
    From a list of tokens, build:
      - audio_plan: list of either tokens (string) or ('FUSE', left_token, right_token)
      - transcript_tokens: list of output tokens for metadata (with 'X ing' -> 'Xing', usa/gib spelled)
    """
    audio_plan = []
    transcript_tokens = []
    i = 0
    while i < len(words):
        w = words[i]
        # consider "X ing" ligature
        if i + 1 < len(words):
            nxt = words[i + 1]
            if (nxt in LIGATURE_SUFFIXES and w in wav_map and nxt in wav_map):
                # Make "Xing" in transcript
                transcript_tokens.append(f"{w}{nxt}")
                audio_plan.append(("FUSE", w, nxt))
                i += 2
                continue
        # normal token
        transcript_tokens.append(spell_letters(w))
        audio_plan.append(w)
        i += 1
    return audio_plan, transcript_tokens

def is_number_word(token):
    # crude: "ten", "100", etc.
    return token in NUMBER_WORDS or re.match(r"^\d+$", token)

def get_gap_wav(gap_s):
    gap_ms = int(round(gap_s * 1000))
    gap_wav = os.path.join(TRIM_CACHE_DIR, f"_gap_{gap_ms}ms.wav")
    if not os.path.exists(gap_wav):
        run_ffmpeg([
            "ffmpeg","-y","-f","lavfi","-t",str(gap_s),
            "-i", f"anullsrc=r={TARGET_SR}:cl=mono",
            "-c:a", TARGET_CODEC, gap_wav
        ])
    return gap_wav

# ------------------------------- main ----------------------------------------
wav_map = build_wav_map(SOURCE_DIR)

try:
    with open(SENTENCES_FILE, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
except FileNotFoundError:
    sys.stderr.write(f"Missing {SENTENCES_FILE}\n"); sys.exit(1)

# Validate sentence tokens exist as word WAVs at the raw level
# (we validate BEFORE ligatures; ligatures use the same source WAVs)
valid_entries, errors = [], []
for idx, sentence in enumerate(lines, start=1):
    if not sentence.strip(): 
        continue
    words = sentence.lower().split()
    missing = [w for w in words if w not in wav_map]
    (errors.append((idx, sentence, missing)) if missing else valid_entries.append((idx, sentence)))

if errors:
    sys.stderr.write(f"Error: missing word WAVs in {len(errors)} line(s):\n")
    for line_no, sentence, missing in errors:
        sys.stderr.write(f"  Line {line_no}: “{sentence}” – missing: {', '.join(missing)}\n")
    sys.exit(1)

# Build initial trim cache for all needed base words (including 'ing', 'usa', 'gib', etc.)
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

# A tiny cache for fused ligature WAVs: key=(left,right) -> path
lig_cache = {}

def ensure_ligature(left_token, right_token):
    """Return path to fused WAV for (left_token, right_token) after ensuring it exists."""
    key = (left_token, right_token)
    if key in lig_cache and os.path.exists(lig_cache[key]):
        return lig_cache[key]
    left_path  = trimmed_paths[left_token]
    right_path = trimmed_paths[right_token]
    # deterministic name based on tokens + mtimes to rebuild if sources change
    lh = str(int(os.path.getmtime(left_path)))
    rh = str(int(os.path.getmtime(right_path)))
    base = f"{LIGATURE_CACHE_PREFIX}{sanitize_for_filename(left_token)}+{sanitize_for_filename(right_token)}_{lh}_{rh}.wav"
    out_path = os.path.join(TRIM_CACHE_DIR, base)
    fuse_two_with_acrossfade(left_path, right_path, out_path, max(LIGATURE_OVERLAP_S, OVERLAP_S))
    lig_cache[key] = out_path
    return out_path

os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_lines = []
for out_idx, (line_no, sentence) in enumerate(valid_entries, start=1):
    words = sentence.lower().split()
    audio_plan, transcript_tokens = plan_ligatures_and_transcript(words, wav_map)
    pace_mult = random.uniform(*SENTENCE_PACE_RANGE)

    # Build [word, gap, word, gap, ...] sequence
    seg_paths = []
    for i, item in enumerate(audio_plan):
        if isinstance(item, tuple) and item and item[0] == "FUSE":
            _, left_tok, right_tok = item
            seg_paths.append(ensure_ligature(left_tok, right_tok))
            this_token = f"{left_tok}{right_tok}"
        else:
            seg_paths.append(trimmed_paths[item])
            this_token = item

        # Add a gap *after* every word except the last
        if i < len(audio_plan) - 1:
            base_gap = 0.045
            jitter = random.uniform(-PACING_JITTER, PACING_JITTER)
            gap = base_gap + jitter
            if (this_token in SEMANTIC_PAUSE_WORDS) or is_number_word(this_token):
                gap += random.uniform(0.18, 0.44)
            if random.random() < 0.16:
                gap += random.uniform(0.10, 0.24)
            gap = max(0.0, gap * pace_mult)
            if gap >= 0.008:
                gap_wav = get_gap_wav(gap)
                seg_paths.append(gap_wav)

    out_wav = os.path.join(OUTPUT_DIR, f"vox_{out_idx:04d}.wav")

    # Always crossfade between each pair in seg_paths (words and gaps both)
    if len(seg_paths) == 1:
        run_ffmpeg([
            "ffmpeg","-y","-i", seg_paths[0],
            "-af", f"areverse,afade=t=in:d={SENTENCE_TAIL_FADE_S},areverse",
            "-ar", str(TARGET_SR), "-ac", str(CHANNELS), "-c:a", TARGET_CODEC, out_wav
        ])
    else:
        # Crossfade between each item in the sequence (word->gap->word->gap...)
        inputs = []
        for p in seg_paths:
            inputs += ["-i", p]
        steps = []
        for i in range(len(seg_paths) - 1):
            left  = "[0:a]" if i == 0 else f"[a{i:02d}]"
            right = f"[{i+1}:a]"
            out   = f"[a{i+1:02d}]"
            # You may want a slightly different fade time for word→gap and gap→word, but just use the same for now
            d_pair = max(0.01, random.uniform(0.030, 0.090)) * pace_mult
            steps.append(f"{left}{right}acrossfade=d={d_pair}:c1={CROSSFADE_C1}:c2={CROSSFADE_C2}{out}")

        last_label = f"[a{len(seg_paths)-1:02d}]"
        steps.append(f"{last_label}areverse,afade=t=in:d={SENTENCE_TAIL_FADE_S},areverse[aout]")
        filt = ";".join(steps)
        run_ffmpeg([
            "ffmpeg","-y", *inputs,
            "-filter_complex", filt, "-map", "[aout]",
            "-ar", str(TARGET_SR), "-ac", str(CHANNELS), "-c:a", TARGET_CODEC, out_wav
        ])

    csv_text = " ".join(transcript_tokens)
    csv_lines.append(f"{OUTPUT_DIR}/vox_{out_idx:04d}.wav|{csv_text}")

with open(CSV_FILE,"w",encoding="utf-8") as out:
    out.write("\n".join(csv_lines))

print(f"✔ Generated {len(valid_entries)} sentence WAVs "
      f"({'overlap ' + str(int(OVERLAP_S*1000)) + 'ms' if OVERLAP_S>0 else 'variable gap mode'}) "
      f"with naturalized pacing and wrote '{CSV_FILE}'.")
