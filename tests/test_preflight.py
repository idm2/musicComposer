import pytest

from songcomposer.generate.preflight import banned_terms, preflight, validate_request
from songcomposer.generate.provider import GENERIC_LIMITS, GenerationRequest
from songcomposer.models import SourceInfo, SpecSection


def req(**over):
    base = dict(
        title="Glass Hour", style_prompt="sparse indie folk, melancholy", negative_style="",
        vocal_gender="any", target_duration_s=120, n_takes=3,
        sections=[
            SpecSection(name="Verse 1", duration_s=60, lines=[
                "The kettle clicks off in the dark", "Your coat still hangs behind the door",
                "I count the streetlights to the park", "And lose my place at twenty-four"]),
            SpecSection(name="Chorus", duration_s=60, lines=[
                "Glass hour, hold me still", "Glass hour, against my will",
                "Turn the morning down", "Until you come around"]),
        ])
    base.update(over)
    return GenerationRequest(**base)


def problems(r, ref=(), banned=()):
    return validate_request(r, GENERIC_LIMITS, list(ref), list(banned))


def test_clean_request_passes():
    assert problems(req()) == []


@pytest.mark.parametrize("line,rule", [
    ("Hold me <break/> still", "markup"),
    ("Hold me [softly] still", "markup"),
    ("Broken � char", "replacement-char"),
    ("soft\xadhyphen line", "soft-hyphen"),
    ("TODO write this line", "placeholder"),
    ("Lyrics go here", "placeholder"),
    ("", "empty-line"),
    ("x" * 201, "line-too-long"),
])
def test_bad_lines_are_caught(line, rule):
    r = req()
    r.sections[0].lines[1] = line
    assert any(p.startswith(f"Verse 1 line 2: {rule}") for p in problems(r)), problems(r)


def test_structure_rules():
    assert any("title" in p for p in problems(req(title="")))
    assert any("title" in p for p in problems(req(title="x" * 81)))
    assert any("style_prompt" in p for p in problems(req(style_prompt=" ")))
    assert any("at least 2 sections" in p for p in problems(req(sections=req().sections[:1])))
    assert any("no lyrics" in p.lower() for p in problems(req(sections=[
        SpecSection(name="A", lines=[], duration_s=60), SpecSection(name="B", lines=[], duration_s=60)])))


def test_duration_rules():
    r = req()
    r.sections[0].duration_s = 2
    assert any("Verse 1: duration" in p for p in problems(r))
    assert any("sum to" in p for p in problems(req(target_duration_s=300)))


def test_crammed_lyrics_caught_by_words_per_second():
    r = req()
    r.sections[0].lines = ["word " * 12] * 25          # 300 words in 60 s = 5 wps
    assert any("words/sec" in p for p in problems(r))


def test_style_must_not_name_the_reference_or_say_in_the_style_of():
    assert any("banned term" in p for p in problems(req(style_prompt="folk like Bon Iver"), banned=["Bon Iver"]))
    assert any("style of" in p for p in problems(req(style_prompt="folk in the style of someone")))


def test_lines_copied_from_the_reference_are_caught():
    ref = ["and", "i", "count", "the", "streetlights", "to", "the", "park", "tonight"]
    out = problems(req(), ref=ref)
    assert any("copies the reference" in p and "streetlights" in p for p in out)


@pytest.mark.parametrize("field", ["title", "style_prompt", "negative_style"])
@pytest.mark.parametrize("text,rule", [
    ("Hold me <break/> still", "markup"),
    ("Broken � char", "replacement-char"),
    ("TODO write this line", "placeholder"),
])
def test_prompt_fields_get_pattern_checks(field, text, rule):
    r = req(**{field: text})
    assert any(p.startswith(f"{field}: {rule}") for p in problems(r)), problems(r)


def test_negative_style_banned_term_and_style_of():
    assert any("banned term" in p for p in problems(req(negative_style="not like Bon Iver"), banned=["Bon Iver"]))
    assert any("style of" in p for p in problems(req(negative_style="in the style of someone")))


def test_negative_style_length_limit():
    over_limit = "x" * (GENERIC_LIMITS.max_negative_chars + 1)
    assert any(p.startswith("negative_style:") and "chars > limit" in p for p in problems(req(negative_style=over_limit)))


def test_suno_negative_style_limit_is_200_not_the_style_limit():
    # Kie.ai rejected a 237-char negativeStyle with 422 after pre-flight had passed it against the 1000-char style limit.
    from songcomposer.generate.suno import SunoProvider
    assert validate_request(req(negative_style="x" * 200), SunoProvider.limits, [], []) == []
    over = validate_request(req(negative_style="x" * 201), SunoProvider.limits, [], [])
    assert any(p == "negative_style: 201 chars > limit 200" for p in over)


def test_negative_style_edm_autotune_still_passes():
    assert problems(req(negative_style="edm, autotune")) == []


def test_banned_terms_from_source_title():
    src = SourceInfo(origin="u", kind="url", title="Bon Iver - Holocene (Official Video)", uploader="Bon Iver",
                     duration_s=1, sample_rate=44100, sha1="a" * 40, ingested_at="now")
    assert banned_terms(src) == ["Bon Iver", "Holocene"]
    assert banned_terms(None) == []


def test_banned_terms_splits_on_hashtags_in_a_real_youtube_shorts_title():
    """I3: the project's own reference (songs/in-the-flow/00-source.json) has a title where hashtags
    run together with no separating space — the artist tag 'cinek' hides inside one long,
    never-matchable term unless '#' is also a delimiter."""
    src = SourceInfo(
        origin="u", kind="url",
        title="No one can beat you\U0001f495 #trending #love#cinek #floating#song#concert "
              "#obessed#shorts#beats#singer#baby",
        uploader="Favori Videolarim ",
        duration_s=1, sample_rate=44100, sha1="a" * 40, ingested_at="now")
    terms = banned_terms(src)
    assert "cinek" in terms

    style_prompt_with_artist_name = req(style_prompt="acoustic folk featuring cinek's voice")
    assert any("cinek" in p for p in problems(style_prompt_with_artist_name, banned=terms))


def test_hashtag_stoplist_drops_generic_words_but_keeps_artist_tag_and_uploader():
    """The over-block found in live use: the project's real reference title yielded 13 banned
    terms including generic words like 'love' and 'song', which then rejected any legitimate
    song mentioning them. Only the genuine artist tag ('cinek') and the uploader phrase
    ('Favori Videolarim') should survive."""
    src = SourceInfo(
        origin="u", kind="url",
        title="No one can beat you\U0001f495 #trending #love#cinek #floating#song#concert "
              "#obessed#shorts#beats#singer#baby",
        uploader="Favori Videolarim ",
        duration_s=1, sample_rate=44100, sha1="a" * 40, ingested_at="now")
    terms = banned_terms(src)
    assert "cinek" in terms
    assert "Favori Videolarim" in terms
    for generic in ("love", "song", "beats", "baby"):
        assert generic not in terms


def test_hashtag_stoplist_still_blocks_style_prompt_naming_the_real_artist():
    src = SourceInfo(
        origin="u", kind="url",
        title="No one can beat you\U0001f495 #trending #love#cinek #floating#song#concert "
              "#obessed#shorts#beats#singer#baby",
        uploader="Favori Videolarim ",
        duration_s=1, sample_rate=44100, sha1="a" * 40, ingested_at="now")
    terms = banned_terms(src)
    r = req(style_prompt="acoustic pop featuring cinek's voice")
    assert any("cinek" in p for p in problems(r, banned=terms))


def test_new_song_with_generic_love_and_song_words_now_passes_preflight():
    """The false positive this fix exists for: retitling the song 'Love Like the Sun Loves'
    with a style prompt describing a love song must not trip the banned-terms check against
    the hashtag-heavy reference title above."""
    src = SourceInfo(
        origin="u", kind="url",
        title="No one can beat you\U0001f495 #trending #love#cinek #floating#song#concert "
              "#obessed#shorts#beats#singer#baby",
        uploader="Favori Videolarim ",
        duration_s=1, sample_rate=44100, sha1="a" * 40, ingested_at="now")
    terms = banned_terms(src)
    r = req(title="Love Like the Sun Loves",
            style_prompt="tender modern acoustic R&B pop love song, 88 BPM, A major")
    assert problems(r, banned=terms) == []


def test_hashtag_stoplist_does_not_apply_to_dash_pipe_split():
    """The stop-list must only apply to terms produced by splitting on '#'. A real artist name
    sitting in a dash/pipe-delimited title (the more likely place to find one) must survive
    even if it happens to collide with a stop-list word."""
    src = SourceInfo(origin="u", kind="url", title="Baby - Some Song", uploader=None,
                     duration_s=1, sample_rate=44100, sha1="a" * 40, ingested_at="now")
    terms = banned_terms(src)
    assert "Baby" in terms


def test_preflight_exits_and_says_nothing_was_generated(capsys):
    with pytest.raises(SystemExit) as e:
        preflight(req(title=""), GENERIC_LIMITS, [], [])
    assert e.value.code == 1
    assert "NOTHING was generated" in capsys.readouterr().out


def test_preflight_passes_quietly(capsys):
    preflight(req(), GENERIC_LIMITS, [], [])
    assert "pre-flight passed" in capsys.readouterr().out


def test_section_style_notes_are_validated_like_a_prompt():
    def with_notes(notes):
        r = req(); r.sections[0].style_notes = notes; return r
    assert problems(with_notes("guitar only, intimate")) == []
    assert any("style_notes: markup" in p for p in problems(with_notes("drop [beat]")))
    assert any("style_notes" in p and "banned term" in p for p in problems(with_notes("like Bon Iver"), banned=["Bon Iver"]))
    assert any("style_notes" in p and "names another work" in p for p in problems(with_notes("sounds like that one hit")))
