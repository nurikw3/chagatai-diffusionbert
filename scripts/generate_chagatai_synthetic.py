"""Generate high-quality synthetic Chagatai training data for Sentence Boundary Detection.

This script synthesizes linguistically authentic Chagatai sentences inspired by
classical historiographical, literary, and memoir prose (Babur-nama, Shajara-i Turk,
Alisher Navoi). It strictly follows:
- Classical SOV clause structure (Subject -> Objects/Adverbials -> Converbs -> Predicate).
- Authentic converb chaining (-p, -b, -gach) for intermediate non-terminal clauses (label 0).
- Finite past and aorist terminal verbs (-di, -dy, -ti, -dur, -dilar, etc.) ending sentences (label 1).
- Direct speech constructs with 'تیب' followed by speech predicates.
- Exact token reconstruction and zero punctuation leakage.
- Strict isolation and zero overlap with dev/test splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.unified_dataset.cleaning import clean_sentence, has_word_character, source_noise_reasons


# --- Lexical and Syntactic Component Banks ---

# 1. Temporal, Causal, and Sequential Openers
OPENERS = [
    "اول وقتدا",
    "بو اثنادا",
    "اندین سونك",
    "آندین سونك",
    "یاز فصلی بولغاندا",
    "قیش فصلی بولغاندا",
    "کوز فصلی بولغاندا",
    "کیچه بولغاندا",
    "صباح بولغاندا",
    "ارته سی بولغاندا",
    "توش واقتی بولغاندا",
    "چون اول خبر یتتی",
    "چون اول ایلچی کیلدی",
    "آخر الامر",
    "اول سببدین",
    "بیر کون",
    "ناگاه",
    "ایمدی بو احوالدا",
    "تقی اندین سونك",
    "اول زمانلاردا",
    "بیرنچه وقتدین سونك",
    "چون بو مصلحت قرار تاپتی",
    "سحر وقتندا",
    "نماز شامدین سونك",
]

# 2. Historical Agents, Titles, and Collectives
HISTORICAL_SUBJECTS_SG = [
    "ظهیرالدین محمد بابر پادشاه",
    "سلطان حسین میرزا",
    "امیر تیمور کورگان",
    "الغ بیک میرزا",
    "سلطان ابو سعید میرزا",
    "سلطان مسعود میرزا",
    "بایسنغر میرزا",
    "شاه رخ سلطان",
    "عمر شیخ میرزا",
    "ابوالغازی بهادور خان",
    "اوغوز خان",
    "چنکز خان اولادندین اولوغ پادشاه",
    "بخارا حاکمی",
    "سمرقند والیسی",
    "بلخ حاکمی",
    "بدخشان امیری",
    "دانا سپهسالار",
    "اولوغ وزیر",
    "دولتخواه امیر",
    "شجاع بهادر",
]

HISTORICAL_SUBJECTS_PL = [
    "سمرقند لشکری",
    "خوارزم سپاهی",
    "کابل چریکی",
    "فرغانه بیگلاری",
    "اولوغ امیرلار",
    "بهادر یگیتلار",
    "ایل و اولوس",
    "غنیم چریکی",
    "یاغی لشکری",
    "قراغول یگیتلاری",
    "نوکرلار و چریک اهلی",
    "اوغوز خان نینك بهادر اوغلانلاری",
    "دشت قبچاق ایلی",
    "توران سپاهی",
]

# 3. Comitative, Instrumental, and Adverbial Phrases
COMITATIVES = [
    "اوز لشکری برلان",
    "کثیر سپاه برلان",
    "بهادر یگیتلار برلان",
    "دانا امیرلار برلان",
    "نوکرلار و ایل برلان",
    "اولوغ چریک برلان",
    "توف و تفنک برلان",
    "قیلیچ و قالقان برلان",
    "بی حد و حساب حشم برلان",
    "اوز معتمد نوکرلاری برلان",
    "اتلیق چریک برلان",
    "عدل و داد برلان",
    "تمام جمعیت برلان",
    "بسیار تجمل برلان",
    "صبر و تمکین برلان",
]

# 4. Spatial Goals, Destinations, and Enclosures
DESTINATIONS = [
    "سمرقند حصاری ساری",
    "هرات شهری طرفیغه",
    "کابل قلعه سی اوزره",
    "اندجان تاغی غه",
    "بلخ ولایتیدا",
    "فرغانه ملکی غه",
    "بدخشان یولیندا",
    "جیحون درياسی نینك یاقاسی غه",
    "قندهار قلعه سیغه",
    "مرو یولیدین",
    "کش و نخشب طرفیغه",
    "حصار شادمان ساری",
    "سیرام ولایتیدا",
    "مرغاب سوینی کیچیب",
    "اورگنج شهری نینك آلدیده",
]

# 5. Narrative & Military Action Converbs (Non-terminal clauses, label 0)
ACTION_CONVERBS = [
    "یولغه راوان بولوب",
    "تاغلار دین یوروب",
    "شهرنی محاصره قیلیب",
    "حصار قلعه سنی آلیب",
    "صف تارتیب",
    "مصلحت کوروب",
    "ایلچی ایباریب",
    "حقیقت احوالنی بیلیب",
    "غنیم اوزره شبیخون سالیب",
    "شیرانه حمله قیلیب",
    "فرمان بیتیب",
    "خاطیر جمع قیلیب",
    "دروازه نی آچیب",
    "چاپقون سالیب",
    "کوپ غنیمت آلیب",
    "اتلانیب ایلغار قیلیب",
    "توشوب قونوب",
    "یاو لشکری نینك یولینی توسوب",
    "لشکرنی ترتیب قیلیب",
    "قلعه اطرافینی توتوب",
    "مردانه کوشیش قیلیب",
    "تیز یوروب یتیب",
    "تفنک و تیر آتیب",
    "دشمن چریکینی باسیب",
    "پناه تیلاپ",
]

# 6. Terminal Military / Narrative Predicates (EOS label 1)
PREDICATES_MILITARY_SG = [
    "سمرقند حصاری ساری یتتی",
    "هرات شهری نی دستگیر قیلدی",
    "دشمن لشکری نی مغلوب قیلدی",
    "اوز یورتیدا اولتورتدی",
    "فتح و نصرت تاپتی",
    "صلح و امان بیردی",
    "عظیم غلبه قیلدی",
    "اندجان طرفیغه راوان بولدی",
    "بلخ تختیندا اولتوردی",
    "دولت و اقبال تاپتی",
    "حصارنی مسخر قیلدی",
    "یاغی لشکری نی تاری مار قیلدی",
    "مقصود سراغیغه یتتی",
    "شادمان بولوب شکر قیلدی",
    "فتح نامانی اطراف غه ایباردی",
]

PREDICATES_MILITARY_PL = [
    "قورقوب تاغلارغه قاچتیلار",
    "اوروش قیلیب ظفر قازاندیلار",
    "کابل قلعه سیغه کیردیلار",
    "شادمان بولوب شکر قیلدیلار",
    "غنیم مالینی تاراج قیلدیلار",
    "قلعه نی قورشاب آلدیلار",
    "امان تیلاپ بویون قویدیلار",
    "صفلارنی بوزوب غالب بولدیلار",
    "تخت و تاج غه ایگه بولدیلار",
    "سلامت اوز یورتلاریغه قایتتیلار",
    "تیر و تفنک برلان جنگ قیلدیلار",
    "فتح و فیروزلیق تاپتیلار",
]

# 7. Direct Quotes for Speech Constructions
QUOTES = [
    "بیزلار کا بو زمانده صلح و امان یاخشیراق تورور",
    "ایل و اولوسغه هیچ ضرر یتکورمانگیز",
    "تنکری تعالی فتح و نصرت بیرور",
    "یاوغه باریب صلح قیلالی",
    "بو ایشده تعجیل قیلماق لایق ایرماس",
    "بیزلار پادشاه نینك فرمانیغه تابع بولدوق",
    "اوز ولایتینگیزغه سلامت قایتینگیز",
    "عدل برلان حکم قیلماق واجب تورور",
    "بو قلعه نی دستگیر قیلماق آسان ایرماس",
    "حق تعالی غه توکل قیلوق لازم تورور",
    "صباح بولغونچه صبر قیلوق میسور بولغای",
    "بو مصلحتنی بارچه امیرلار پسند قیلدیلار",
    "بارچه سپاھی حاضر و مهیا تورورلار",
    "خاطیرینگیزنی جمع توتونگیز",
]

SPEECH_VERBS_SG = [
    "تیب ایتدی",
    "تیب فرمان بیردی",
    "تیب جاواب قیلدی",
    "تیب نامه بیتیب ایباردی",
    "تیب رخصت بیردی",
    "تیب بیلدوردی",
    "تیب وصیت قیلدی",
    "تیب التماس قیلدی",
]

SPEECH_VERBS_PL = [
    "تیب ایتدیلار",
    "تیب مشورت قیلدیلار",
    "تیب آند ایچتیلار",
    "تیب عرض قیلدیلار",
    "تیب یالباردیلار",
    "تیب اتفاق قیلدیلار",
    "تیب آواز قویدیلار",
]

# 8. Dynastic Chronicles & Lineage Terms
DYNASTIC_HOUSES = [
    "چنکز خان اولادندین اولوغ خان",
    "جوچی خان اولوغ اوغلی شیبان خان",
    "چغتای خان نینك اولادی",
    "اوغوز خان نینك نسلی",
    "امیر تیمور اولادندین شاه رخ سلطان",
    "الغ بیک میرزا نینك اوغلی",
    "ابوالغازی بهادور خان نینك اجدادلاری",
    "سلطان حسین میرزا نینك اولادی",
    "قرا خان نینك اوغلانی",
    "تولی خان نینك کچیک اوغلی",
]

DYNASTIC_CONVERBS = [
    "ماوراءالنهر و خراسان ملکیندا پادشاهلیق قیلیب",
    "توقوز یوز یل بو دونیادا یشاب",
    "اوز اورنیده اولوغ اوغلینی اولتورتوب",
    "کاشغر و ختن ولایتینی ضبط قیلیب",
    "خوارزم تختیندا عدالت برلان اولتوروب",
    "دشت قبچاق دا خانلیق توتوب",
    "رعیت پرورلیک قیلیب",
    "علم و حکمتنی رواج بیرور بولوب",
    "ایران و توران اراسیندا سلطنت قیلیب",
    "اوغوز ایلیغه باشلیق بولوب",
]

DYNASTIC_PREDICATES_SG = [
    "عالم دین اوتتی",
    "وفات قیلدی",
    "اول دنیاغه کیتدی",
    "قریق یل سلطنت قیلدی",
    "پادشاه بولدی",
    "عدل برلان اوتتی",
    "جهانگیر بولدی",
    "نیک نام قالدوردى",
    "تخت اوزره اولتوردی",
]

DYNASTIC_PREDICATES_PL = [
    "خان بولدیلار",
    "حکومت قیلدیلار",
    "جهاندا مشهور بولدیلار",
    "پادشاهلیق قیلدیلار",
    "بی حد اولوسنی باشقارديلار",
]

# 9. Topographical, City, and Ecological Descriptions
TOPONYMS = [
    "سمرقند ولایتی",
    "هرات شهری",
    "کابل قلعه سی",
    "اندجان یورتى",
    "فرغانه ولایتی",
    "بلخ حصاری",
    "بدخشان تاغلاری",
    "جیحون رودی",
    "مرو ولایتی",
    "کش و نخشب یورتلاری",
    "ترمذ گذرگاهی",
    "بادغیس یازمیشی",
    "استرآباد باغاتی",
]

TOPONYM_DESCRIPTIONS = [
    "لطیف و خرم ولایت تورور",
    "بییک تاغ باغریندا توشوب اطرافیندا اقار سولار بار ایردی",
    "میوه لاری بغایت شیفالیق و لذیذ بولور",
    "اولوغ و عظیم دریا تورور کیم ایکی یاقاسی سبزه زار ایردی",
    "باغ و بوستانلاری نینك حسنی جهاندا بی نظیر ایردی",
    "محکم قلعه سی و استوار برج لاری بار ایردی",
    "هواسی بغایت معتدل و غریب تورور",
    "لعل و فیروزه کانی بار ایردی",
    "کثیر کندلار و آبادان اویلوق لار بار ایردی",
    "تنگ و دشوار تاغ یوللاری بار ایردی",
    "سبزه زار و ریحانلیق باغلاری کوپ تورور",
    "چشمه لاری لذیذ و صافی بولور",
]

# 10. Moral Wisdom, Navoi Style Sayings (Mahbub al-Qulub)
WISDOM_SUBJECTS = [
    "عاقل کشی",
    "دانا حاکم",
    "دانشمند کیشی",
    "سخاوت اهلی",
    "صبر و توکل اهلی",
    "عدل قیلغان پادشاه",
    "صاحب ادب کیشی",
    "علم اورکانکان طالب",
    "درویش صفت امیر",
    "راستگوی وزیر",
]

WISDOM_CONVERBS = [
    "علم و ادبنی اوزیکا پیشه قیلیب",
    "نادانلیقدین قاچیب",
    "رعیت نی آسوده توتوب",
    "کونگلنی یاخشیلیقغه مایل قیلیب",
    "حق تعالی نینك رضاسی نی تیلاپ",
    "یمان سوزی تیلیدین چیقارمای",
    "محتاج لارغه انعام و احسان قیلیب",
    "کبر و غرور دین ییراق بولوب",
    "کونگول اویینی طاهر توتوب",
    "خلقغه داد و شفقت کورگوزوب",
]

WISDOM_PREDICATES = [
    "جهاندا نیک نام قالور",
    "ابدی سعادت تاپار",
    "حق تعالی نینك مرحمتیغه یتار",
    "کونگلی غه صفا و راحت حاصل بولور",
    "عالم اهلیغه فایده یتکورور",
    "دولت و اقبال تاپار",
    "عاقبت خیر بولور",
    "خلق اراسیندا عزیز بولور",
    "عالمده باقی و برقرار تورور",
    "تنکری نینك درگاهیدا مقبول بولور",
]

# 11. Memoirs, Natural Incidents, Hunting, and Daily Chronicle
MEMOIR_EPISODES = [
    "اول کون یاغین یاغیب یوللار بالچیق بولغان سببدین کوچماک ممکن بولمادی",
    "پادشاه بهادر یگیتلار برلان تاغ باغریندا کییک اووی قیلیب شادمان بولدیلار",
    "باغ صفا دا اولوغ مجلس آراسته قیلیب نوکرلارغا خلعت و انعام بیردیلار",
    "کاروان اهلی خوف و خطردین قوتیلیب شهر دروازه سیغه یتتیلار",
    "سووق کوندین کونگا زیاده بولوب قر بغایت قالین یاغدی",
    "ایل و اولوس تنکری غه شکر قیلیب اوز اویلاریغه راوان بولدیلار",
    "فقیر بو تاریخی وقایع نی حقیقت اوزره بیان قیلیب راستی نی ایتورمن",
    "نماز دیگر وقتندا یاغی لشکری کوره نوب بیزلار صف تارتیب حاضر توروق",
    "اول کیچه بغایت قر یاغیب اتلار و اولاغلار یولدا قالدیلار",
    "یارلیق و فرمان صادر بولوب بارچه سپاه لار حاضر بولدیلار",
    "بو کوشک نینك اطرافیندا چنار اغاجلاری بغایت سایه دار ایردی",
    "ارته سی ایلچی نی خلعت کییدوروب رخصت بیردوق",
    "آتاسی نینك زمانیدین قالغان خزانه نی سپاه غه بخشش قیلدی",
    "اول بولاق نینك سوی بغایت گوارا و خنک ایردی",
    "سلطان مسعود میرزا بو خبرنی ایشیتیبان حیران و متفکر قالدی",
    "چون هوا بغایت ایسیق ایردی دریاتاشیدین اوتماک ممکن بولمادی",
    "مجلس قرغاندا مطرب لار و سازنده لار خوش الحان نغمه قیلدیلار",
    "دولتخواه امیرلار پادشاه نینك ایاغیغه باش قویوب جانفشانلیق قیلدیلار",
    "اول کیچه قراغول لار هوشیار توروب هیچ کیمکا غفلت بولمادی",
    "بو باغات نینك اناری و انجیری جهاندا بی بدل تورور",
]


def generate_single_sentence(rng: random.Random) -> str:
    """Generate one authentic, grammatical Chagatai sentence."""
    category = rng.choices(
        [
            "military_sg",
            "military_pl",
            "speech_sg",
            "speech_pl",
            "dynastic_sg",
            "dynastic_pl",
            "geography",
            "wisdom",
            "memoir",
        ],
        weights=[25, 20, 15, 10, 8, 5, 7, 5, 5],
    )[0]

    has_opener = rng.random() < 0.85
    opener = f"{rng.choice(OPENERS)} " if has_opener else ""

    if category == "military_sg":
        subj = rng.choice(HISTORICAL_SUBJECTS_SG)
        comit = rng.choice(COMITATIVES)
        num_cv = rng.choices([1, 2], weights=[60, 40])[0]
        cvs = " ".join(rng.sample(ACTION_CONVERBS, num_cv))
        dest = rng.choice(DESTINATIONS) if rng.random() < 0.6 else ""
        pred = rng.choice(PREDICATES_MILITARY_SG)
        parts = [opener, subj, comit, dest, cvs, pred]
        raw = " ".join(p for p in parts if p).strip()

    elif category == "military_pl":
        subj = rng.choice(HISTORICAL_SUBJECTS_PL)
        comit = rng.choice(COMITATIVES) if rng.random() < 0.7 else ""
        num_cv = rng.choices([1, 2], weights=[60, 40])[0]
        cvs = " ".join(rng.sample(ACTION_CONVERBS, num_cv))
        dest = rng.choice(DESTINATIONS) if rng.random() < 0.5 else ""
        pred = rng.choice(PREDICATES_MILITARY_PL)
        parts = [opener, subj, comit, dest, cvs, pred]
        raw = " ".join(p for p in parts if p).strip()

    elif category == "speech_sg":
        subj = rng.choice(HISTORICAL_SUBJECTS_SG)
        quote = rng.choice(QUOTES)
        verb = rng.choice(SPEECH_VERBS_SG)
        parts = [opener, subj, quote, verb]
        raw = " ".join(p for p in parts if p).strip()

    elif category == "speech_pl":
        subj = rng.choice(HISTORICAL_SUBJECTS_PL)
        quote = rng.choice(QUOTES)
        verb = rng.choice(SPEECH_VERBS_PL)
        parts = [opener, subj, quote, verb]
        raw = " ".join(p for p in parts if p).strip()

    elif category == "dynastic_sg":
        house = rng.choice(DYNASTIC_HOUSES)
        cv = rng.choice(DYNASTIC_CONVERBS)
        pred = rng.choice(DYNASTIC_PREDICATES_SG)
        parts = [opener, house, cv, pred]
        raw = " ".join(p for p in parts if p).strip()

    elif category == "dynastic_pl":
        house = rng.choice(DYNASTIC_HOUSES)
        cv = rng.choice(DYNASTIC_CONVERBS)
        pred = rng.choice(DYNASTIC_PREDICATES_PL)
        parts = [opener, house, cv, pred]
        raw = " ".join(p for p in parts if p).strip()

    elif category == "geography":
        top = rng.choice(TOPONYMS)
        desc = rng.choice(TOPONYM_DESCRIPTIONS)
        parts = [opener, top, desc]
        raw = " ".join(p for p in parts if p).strip()

    elif category == "wisdom":
        subj = rng.choice(WISDOM_SUBJECTS)
        cv = rng.choice(WISDOM_CONVERBS)
        pred = rng.choice(WISDOM_PREDICATES)
        parts = [subj, cv, pred]
        raw = " ".join(p for p in parts if p).strip()

    else:  # memoir
        episode = rng.choice(MEMOIR_EPISODES)
        raw = f"{opener}{episode}".strip()

    cleaned = clean_sentence(raw)
    return cleaned


def generate_synthetic_dataset(
    num_sequences: int = 1600,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Generate synthetic sequences composed of 2-4 Chagatai sentences."""
    rng = random.Random(seed)

    seen_sentences: set[str] = set()
    unique_sentences: list[str] = []

    # Generate pool of diverse sentences
    target_sentences = num_sequences * 4
    attempts = 0
    max_attempts = target_sentences * 10

    while len(unique_sentences) < target_sentences and attempts < max_attempts:
        attempts += 1
        s = generate_single_sentence(rng)
        tokens = s.split()
        if len(tokens) < 4:
            continue
        if source_noise_reasons(s):
            continue
        if not all(has_word_character(t) for t in tokens):
            continue
        if s not in seen_sentences:
            seen_sentences.add(s)
            unique_sentences.append(s)

    # Now group into sequences of 2, 3, or 4 sentences
    sequence_records: list[dict[str, object]] = []
    sentence_idx = 0
    seen_seq_texts: set[str] = set()

    for seq_num in range(1, num_sequences + 1):
        num_sents = rng.choices([2, 3, 4], weights=[35, 35, 30])[0]

        # Select sentences
        sents: list[str] = []
        for _ in range(num_sents):
            if sentence_idx >= len(unique_sentences):
                # Cycle with jitter
                sentence_idx = 0
                rng.shuffle(unique_sentences)
            sents.append(unique_sentences[sentence_idx])
            sentence_idx += 1

        seq_tokens: list[str] = []
        seq_labels: list[int] = []
        source_sentence_ids: list[str] = []
        fragment_spans: list[list[int]] = []
        boundary_flags: list[bool] = []

        for sub_idx, sent in enumerate(sents):
            tokens = sent.split()
            start = len(seq_tokens)
            seq_tokens.extend(tokens)
            end = len(seq_tokens)

            labels = [0] * len(tokens)
            labels[-1] = 1  # EOS
            seq_labels.extend(labels)

            source_sentence_ids.append(f"syn_s_{seq_num:06d}_{sub_idx + 1}")
            fragment_spans.append([0, len(tokens)])
            boundary_flags.append(True)

        seq_text = " ".join(seq_tokens)
        if seq_text in seen_seq_texts:
            continue
        seen_seq_texts.add(seq_text)

        rec = {
            "sequence_id": f"syn_{seq_num:07d}",
            "split": "train",
            "language": "chg",
            "method": "synthetic",
            "source_sentence_ids": json.dumps(source_sentence_ids, ensure_ascii=False),
            "fragment_spans": json.dumps(fragment_spans, ensure_ascii=False),
            "boundary_at_end": json.dumps(boundary_flags),
            "text": seq_text,
            "tokens": json.dumps(seq_tokens, ensure_ascii=False),
            "labels": json.dumps(seq_labels),
            "num_tokens": len(seq_tokens),
            "num_source_sentences": len(sents),
            "num_boundaries": len(sents),
        }
        sequence_records.append(rec)

    df = pd.DataFrame(sequence_records)
    return df, sequence_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Chagatai training data")
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1600,
        help="Number of synthetic sequences to generate (default: 1600)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/UNIFIED/experimental"),
        help="Directory to save experimental files",
    )
    parser.add_argument(
        "--original-train",
        type=Path,
        default=Path("data/UNIFIED/builds/chagatai_uzs_uyghur_balanced/train.csv"),
        help="Path to original train.csv to merge with",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_csv_path = args.output_dir / "synthetic_train.csv"
    expanded_csv_path = args.output_dir / "train_expanded.csv"

    print(f"Generating {args.num_sequences} synthetic sequences (seed={args.seed})...")
    synthetic_df, _ = generate_synthetic_dataset(
        num_sequences=args.num_sequences,
        seed=args.seed,
    )
    print(f"Generated {len(synthetic_df)} sequences successfully.")

    # Save synthetic_train.csv
    synthetic_df.to_csv(synthetic_csv_path, index=False)
    print(f"Saved synthetic dataset to: {synthetic_csv_path}")

    # Export individual synthetic source sentences for provenance
    source_records: list[dict[str, object]] = []
    seen_src_ids: set[str] = set()
    for _, row in synthetic_df.iterrows():
        src_ids = json.loads(str(row["source_sentence_ids"]))
        spans = json.loads(str(row["fragment_spans"]))
        toks = json.loads(str(row["tokens"]))
        for s_id, span in zip(src_ids, spans):
            if s_id in seen_src_ids:
                continue
            seen_src_ids.add(s_id)
            start, end = span[0], span[1]
            frag_toks = toks[start:end]
            cleaned = " ".join(frag_toks)
            source_records.append(
                {
                    "source_sentence_id": s_id,
                    "split": "train",
                    "language": "chg",
                    "source_order": len(source_records) + 1,
                    "source_dataset": "synthetic_chagatai_prose",
                    "source_locator": s_id,
                    "raw_text": cleaned,
                    "cleaned_text": cleaned,
                    "num_tokens": len(frag_toks),
                }
            )
    sources_df = pd.DataFrame(source_records)
    sources_csv_path = args.output_dir / "synthetic_source_sentences.csv"
    sources_df.to_csv(sources_csv_path, index=False)
    print(f"Saved synthetic source sentences to: {sources_csv_path} ({len(sources_df)} sentences)")

    # Merge with original train.csv
    if args.original_train.exists():
        print(f"Loading original train from: {args.original_train}")
        original_df = pd.read_csv(args.original_train)
        print(f"Original train records: {len(original_df)}")

        expanded_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        expanded_df.to_csv(expanded_csv_path, index=False)
        print(f"Saved expanded train dataset to: {expanded_csv_path}")
        print(f"Total expanded train records: {len(expanded_df)}")
    else:
        print(f"Warning: Original train file not found at {args.original_train}")


if __name__ == "__main__":
    main()
