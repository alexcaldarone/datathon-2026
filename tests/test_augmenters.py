import pytest
from app.ingestion.components.augmenters import AnchorsAugmenter, AugmentedFeature, FeatureType
from app.participant.components.utils import Config

NUM_ATTRIBUTES = 16

DAYLIGHT       = 0
ORIENTATION    = 1
NOISE          = 2
SOUND_INSUL    = 3
VIEW           = 4
GREEN          = 5
WALKABILITY    = 6
PUBLIC_TRANSP  = 7
CENTRALITY     = 8
SAFETY         = 9
PROXIMITY      = 10
AIR_QUALITY    = 11
SPACIOUSNESS   = 12
MODERNITY      = 13
LIVABILITY     = 14
WFH            = 15


@pytest.fixture(scope="module")
def augmenter() -> AnchorsAugmenter:
    return AnchorsAugmenter(Config.get_cfg())


def _vec(augmenter: AnchorsAugmenter, query: str) -> list[float]:
    return augmenter.augment({"full_text": query}).content


# ── Shape & bounds ────────────────────────────────────────────────────────────

def test_returns_augmented_feature(augmenter: AnchorsAugmenter) -> None:
    assert isinstance(augmenter.augment({"full_text": "Wohnung in Zuerich"}), AugmentedFeature)

def test_feature_type_is_dense(augmenter: AnchorsAugmenter) -> None:
    assert augmenter.augment({"full_text": "Wohnung in Zuerich"}).type == FeatureType.DENSE

def test_content_is_list_of_16_floats(augmenter: AnchorsAugmenter) -> None:
    content = augmenter.augment({"full_text": "Wohnung in Zuerich"}).content
    assert isinstance(content, list)
    assert len(content) == NUM_ATTRIBUTES
    assert all(isinstance(v, float) for v in content)

def test_all_values_within_bounds(augmenter: AnchorsAugmenter) -> None:
    for q in [
        "Ruhige helle Wohnung mit Seesicht gesucht",
        "Ich will eine laute Wohnung nahe am Nachtleben",
        "Homeoffice, schnelles Internet, Arbeitszimmer",
        "Ich vermeide moderne Wohnungen, lieber Altbau",
        "Familienfreundliches Quartier, sicherer Schulweg",
    ]:
        assert all(-1.0 <= v <= 1.0 for v in _vec(augmenter, q)), f"Out of bounds: {q!r}"


# ── daylight_score ────────────────────────────────────────────────────────────

def test_daylight_pos_lichtdurchflutet(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Helle lichtdurchflutete Wohnung mit grossen Fenstern")[DAYLIGHT] > 0

def test_daylight_pos_viel_tageslicht(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Tageslicht ist mir sehr wichtig, ich arbeite am Fenster")[DAYLIGHT] > 0

def test_daylight_pos_suedseite(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Suedseite mit Sonne den ganzen Nachmittag")[DAYLIGHT] > 0

def test_daylight_pos_morgensonne(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Morgensonne im Schlafzimmer ist ein Traum fuer mich")[DAYLIGHT] > 0

def test_daylight_pos_sonnendurchflutet(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Sonnendurchflutetes Wohnzimmer, viel natuerliches Licht")[DAYLIGHT] > 0

def test_daylight_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Helli Wohnig mit viel Sunne isch mir sehr wichtig")[DAYLIGHT] > 0

def test_daylight_neg_nordseite(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Nordseitige Lage ist kein Problem fuer mich")[DAYLIGHT] < 0

def test_daylight_neg_dunkel_gemuetlich(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Dunkle gemuetliche Wohnung bevorzugt, ich brauche kein helles Apartment")[DAYLIGHT] < 0

def test_daylight_neg_tageslicht_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Tageslicht spielt keine Rolle, ich bin selten zuhause")[DAYLIGHT] < 0

def test_daylight_neg_verdunkelung(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Verdunkelung wichtiger als Helligkeit, ich schlafe tagsue ber")[DAYLIGHT] < 0

def test_daylight_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Es muss noed hell si, mir isch das voellig egal")[DAYLIGHT] < 0

def test_daylight_ordering(augmenter: AnchorsAugmenter) -> None:
    bright = _vec(augmenter, "Sehr helle Wohnung, viel Sonne und Tageslicht")[DAYLIGHT]
    dark   = _vec(augmenter, "Kein Wert auf Sonneneinstrahlung, Nordseite ok")[DAYLIGHT]
    assert bright > dark


# ── orientation_quality ───────────────────────────────────────────────────────

def test_orientation_pos_suedausrichtung(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Suedausrichtung der Wohnung ist mir sehr wichtig")[ORIENTATION] > 0

def test_orientation_pos_westseite(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Westseite fuer schoene Abendsonne bevorzugt")[ORIENTATION] > 0

def test_orientation_pos_ostseite(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ostseite damit ich morgens Sonne ins Schlafzimmer bekomme")[ORIENTATION] > 0

def test_orientation_pos_sonnenseite(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Sonnenseite des Gebaeudes, optimale Ausrichtung zur Sonne")[ORIENTATION] > 0

def test_orientation_pos_suedwestlage(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Suedwestlage bevorzugt, Sonne vom Mittag bis Abend")[ORIENTATION] > 0

def test_orientation_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "D Wohnig soll sunsiitig si, Suedausrichtig isch wichtig")[ORIENTATION] > 0

def test_orientation_neg_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Himmelsrichtung ist mir voellig egal")[ORIENTATION] < 0

def test_orientation_neg_nordseite_ok(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Nordseite oder Suedseite macht mir keinen Unterschied")[ORIENTATION] < 0

def test_orientation_neg_keine_rolle(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Himmelsrichtung ist mir unwichtig, ich lege keinen Wert auf Ausrichtung")[ORIENTATION] < 0

def test_orientation_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ausrichtig isch mer wurscht, egal wie d Wohnig steht")[ORIENTATION] < 0

def test_orientation_neg_nordlage(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Nordlage kein Problem, ich bin tagsue ber im Buero")[ORIENTATION] < 0

def test_orientation_ordering(augmenter: AnchorsAugmenter) -> None:
    south  = _vec(augmenter, "Suedlage damit die Sonne reinkommt, sehr wichtig")[ORIENTATION]
    indiff = _vec(augmenter, "Egal welche Himmelsrichtung, das ist mir unwichtig")[ORIENTATION]
    assert south > indiff


# ── noise_level ───────────────────────────────────────────────────────────────

def test_noise_pos_ruhige_wohnlage(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ruhige Wohnlage ohne Strassenlaerm gesucht")[NOISE] > 0

def test_noise_pos_stille(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Absolute Stille wichtig, ich brauche Ruhe zum Schlafen")[NOISE] > 0

def test_noise_pos_laermgeschuetzt(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Laermgeschuetzte Lage, kein Durchgangsverkehr")[NOISE] > 0

def test_noise_pos_tempo30(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Tempo-30-Zone oder ruhige Seitenstrasse bevorzugt")[NOISE] > 0

def test_noise_pos_kein_flulaerm(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Keine Fluglaerm oder Zugslaerm, ruhiger Innenhof")[NOISE] > 0

def test_noise_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Stilli Umgaebig isch mer sehr wichtig, kei Laerm")[NOISE] > 0

def test_noise_neg_laute_wohnung(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich will eine laute Wohnung nahe am Nachtleben")[NOISE] < 0

def test_noise_neg_laerm_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Strassenlaerm stoert mich nicht, belebte Strasse ist ok")[NOISE] < 0

def test_noise_neg_laut_lebendig(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Laut und lebendig soll es sein, ich mag Betrieb")[NOISE] < 0

def test_noise_neg_nachtleben(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Nachtleben und Belebung sind mir wichtig, Laerm macht nichts")[NOISE] < 0

def test_noise_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Laerm isch kein Problem, ich stoer mich noed dra")[NOISE] < 0

def test_noise_ordering(augmenter: AnchorsAugmenter) -> None:
    quiet = _vec(augmenter, "Ich schlafe schlecht bei Laerm, ruhige Lage ein Muss")[NOISE]
    noisy = _vec(augmenter, "Lautes lebendiges Quartier, Laermpegel spielt keine Rolle")[NOISE]
    assert quiet > noisy


# ── sound_insulation_quality ──────────────────────────────────────────────────

def test_sound_pos_dicke_waende(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Dicke Waende und gute Schalldaemmung sehr wichtig")[SOUND_INSUL] > 0

def test_sound_pos_nachbarn_nicht_hoeren(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Man soll die Nachbarn nicht hoeren koennen")[SOUND_INSUL] > 0

def test_sound_pos_dreifachverglasung(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Dreifachverglasung und schallisolierte Waende gewuenscht")[SOUND_INSUL] > 0

def test_sound_pos_kein_trittschall(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kein Trittschall von oben, akustisch gut gedaemmte Wohnung")[SOUND_INSUL] > 0

def test_sound_pos_lärm_nachbarn_nogo(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Laerm von Nachbarn ist ein absolutes No-Go fuer mich")[SOUND_INSUL] > 0

def test_sound_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Guti Schalldaemmig isch sehr wichtig, Nachbarn noed hoere")[SOUND_INSUL] > 0

def test_sound_neg_schalldaemmung_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Schalldaemmung spielt keine Rolle fuer mich")[SOUND_INSUL] < 0

def test_sound_neg_nachbarn_ok(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Nachbarn hoeren stoert mich nicht, kein Bedarf an Schallschutz")[SOUND_INSUL] < 0

def test_sound_neg_trittschall_ok(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Trittschall ist kein Problem fuer mich, ich bin taub gegenueber Laerm")[SOUND_INSUL] < 0

def test_sound_neg_kein_wert(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich lege keinen Wert auf Schallschutz oder Daemmung")[SOUND_INSUL] < 0

def test_sound_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Schalldaemmig isch mer wurscht, Nachbarn hoere isch ok")[SOUND_INSUL] < 0

def test_sound_ordering(augmenter: AnchorsAugmenter) -> None:
    good = _vec(augmenter, "Ausgezeichnete Schalldaemmung, man hoert absolut nichts")[SOUND_INSUL]
    bad  = _vec(augmenter, "Schalldaemmung ist mir egal, Nachbarn hoeren kein Problem")[SOUND_INSUL]
    assert good > bad


# ── view_quality_score ────────────────────────────────────────────────────────

def test_view_pos_seesicht(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Seesicht sehr wichtig, Blick auf den Zuerichsee")[VIEW] > 0

def test_view_pos_alpenpanorama(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Alpenpanorama vom Balkon, unverbaubare Bergaussicht")[VIEW] > 0

def test_view_pos_stadtblick(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Weitblick ueber die Stadt, Dachterrasse mit Fernblick")[VIEW] > 0

def test_view_pos_bergblick(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Bergblick aus dem Wohnzimmer, schoene Aussicht ist Pflicht")[VIEW] > 0

def test_view_pos_unverbaubar(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Unverbaubare Aussicht, ich moechte keinen Betonklotz vor dem Fenster")[VIEW] > 0

def test_view_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Tolli Ussicht isch mir sehr wichtig, Seesicht oder Bergblick")[VIEW] > 0

def test_view_neg_erdgeschoss(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Erdgeschoss ist voellig in Ordnung, Aussicht egal")[VIEW] < 0

def test_view_neg_aussicht_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich brauche keine schoene Aussicht, Hauptsache guenstig")[VIEW] < 0

def test_view_neg_kein_bedarf(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kein Bedarf an besonderem Ausblick, erstes Stockwerk reicht")[VIEW] < 0

def test_view_neg_achte_nicht(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich achte nicht auf den Ausblick, das ist mir unwichtig")[VIEW] < 0

def test_view_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ussicht isch mer voellig egal, kei schoeni Ussicht noetig")[VIEW] < 0

def test_view_ordering(augmenter: AnchorsAugmenter) -> None:
    great = _vec(augmenter, "Panoramablick auf See und Berge, schoene Aussicht Muss")[VIEW]
    none  = _vec(augmenter, "Aussicht spielt keine Rolle fuer mich, Erdgeschoss ok")[VIEW]
    assert great > none


# ── green_view_ratio ──────────────────────────────────────────────────────────

def test_green_pos_park_nearby(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Park in der Naehe, Gruenflaehen in der Umgebung")[GREEN] > 0

def test_green_pos_wald(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Wald direkt vor der Haustuer, naturnahes Wohnumfeld")[GREEN] > 0

def test_green_pos_gartenblick(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Gartenblick aus dem Fenster, umgeben von Baeumen und Gruen")[GREEN] > 0

def test_green_pos_naherholung(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Naherholungsgebiet in Gehdistanz, Wiese vor der Haustuer")[GREEN] > 0

def test_green_pos_gartenanteil(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Wohnung mit Gartenanteil, umgeben von Baeumen und Gaerten")[GREEN] > 0

def test_green_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Gruene Natur isch mir sehr wichtig, ich will Parks und Wald in dr Naehe")[GREEN] > 0

def test_green_neg_urban(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Urbane Lage wichtiger als Natur, Gruenflaehen nicht noetig")[GREEN] < 0

def test_green_neg_keine_parks(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich brauche keine Parks in der Naehe, bin kein Naturmensch")[GREEN] < 0

def test_green_neg_natur_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Natur spielt keine Rolle fuer mich, Stadtlage bevorzugt")[GREEN] < 0

def test_green_neg_beton_ok(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Innenstadtlage ohne Gruen ist voellig ok fuer mich")[GREEN] < 0

def test_green_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Gruen isch mer egal, Stadtlage ohne Natur isch ok")[GREEN] < 0

def test_green_ordering(augmenter: AnchorsAugmenter) -> None:
    nature = _vec(augmenter, "Gruenes Wohnquartier, viel Natur und Parks in der Naehe")[GREEN]
    urban  = _vec(augmenter, "Betonstadt, kein Bedarf an Gruenflaehen")[GREEN]
    assert nature > urban


# ── walkability_score ─────────────────────────────────────────────────────────

def test_walk_pos_migros_eck(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Migros oder Coop ums Eck, Supermarkt in der Naehe")[WALKABILITY] > 0

def test_walk_pos_alles_zu_fuss(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Alles in Gehdistanz, ich moechte alles zu Fuss erledigen")[WALKABILITY] > 0

def test_walk_pos_baeckerei_apotheke(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Baeckerei, Apotheke und Arzt zu Fuss erreichbar")[WALKABILITY] > 0

def test_walk_pos_lebendiges_quartier(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Lebendiges Quartier mit vielen Laeden und Restaurants")[WALKABILITY] > 0

def test_walk_pos_mitten_im_geschehen(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Mitten im Geschehen wohnen, kurze Wege zu allem")[WALKABILITY] > 0

def test_walk_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Alles z Fuaess erreichbar isch super, Laede und Migros i dr Naeechi")[WALKABILITY] > 0

def test_walk_neg_auto(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich habe ein Auto, Laeden zu Fuss nicht wichtig")[WALKABILITY] < 0

def test_walk_neg_abgelegen(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Abgelegene ruhige Lage bevorzugt, kein Bedarf an Naehe zu Laeden")[WALKABILITY] < 0

def test_walk_neg_einkauf_auto(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich fahre sowieso mit dem Auto einkaufen")[WALKABILITY] < 0

def test_walk_neg_infra_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Infrastruktur im Quartier nicht so wichtig, habe ein Auto")[WALKABILITY] < 0

def test_walk_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Walkability isch mer wurscht, ich bruche kei Laede i dr Naeechi")[WALKABILITY] < 0

def test_walk_ordering(augmenter: AnchorsAugmenter) -> None:
    walk = _vec(augmenter, "Alles in Laufdistanz, Supermarkt um die Ecke")[WALKABILITY]
    car  = _vec(augmenter, "Auto vorhanden, Gehweg zu Laeden egal")[WALKABILITY]
    assert walk > car


# ── public_transport_score ────────────────────────────────────────────────────

def test_pt_pos_tramhaltestelle(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Tramhaltestelle vor der Haustuer, gute OeV-Anbindung")[PUBLIC_TRANSP] > 0

def test_pt_pos_sbahn(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "S-Bahn-Anschluss fusslaeuefig, Bahnhof in Gehdistanz")[PUBLIC_TRANSP] > 0

def test_pt_pos_kein_auto(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich nutze taeglich OeV, Tram und Bus direkt vor der Haustuer wichtig")[PUBLIC_TRANSP] > 0

def test_pt_pos_optimale_anbindung(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Optimale OeV-Anbindung ins Stadtzentrum sehr wichtig")[PUBLIC_TRANSP] > 0

def test_pt_pos_bushaltestelle(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Bushaltestelle direkt vor dem Haus, wenige Minuten zur Haltestelle")[PUBLIC_TRANSP] > 0

def test_pt_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Gueti OeV-Verbindig isch ein Muss, noed bi de Tramstation")[PUBLIC_TRANSP] > 0

def test_pt_neg_auto_vorhanden(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich habe ein Auto, OeV-Anbindung ist mir egal")[PUBLIC_TRANSP] < 0

def test_pt_neg_fahre_auto(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich fahre immer mit dem Auto zur Arbeit")[PUBLIC_TRANSP] < 0

def test_pt_neg_tram_unwichtig(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Tramhaltestelle nicht notwendig, oeffentlicher Verkehr unwichtig")[PUBLIC_TRANSP] < 0

def test_pt_neg_kein_bedarf(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "OeV-Anbindung ist mir voellig unwichtig, ich fahre immer mit dem Auto")[PUBLIC_TRANSP] < 0

def test_pt_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "OeV isch mer voellig egal, ich bruche kei OeV")[PUBLIC_TRANSP] < 0

def test_pt_ordering(augmenter: AnchorsAugmenter) -> None:
    transit = _vec(augmenter, "Taeglich mit Tram, S-Bahn sehr wichtig, kein Auto")[PUBLIC_TRANSP]
    car     = _vec(augmenter, "OeV spielt keine Rolle, fahre immer Auto")[PUBLIC_TRANSP]
    assert transit > car


# ── centrality_index ──────────────────────────────────────────────────────────

def test_central_pos_innenstadt(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Innenstadtlage sehr bevorzugt, mitten in der Stadt")[CENTRALITY] > 0

def test_central_pos_herz_der_stadt(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Im Herzen der Stadt wohnen, urban und zentral")[CENTRALITY] > 0

def test_central_pos_stadtzentrum(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Wohnung im Stadtzentrum gesucht, sehr zentrale Lage")[CENTRALITY] > 0

def test_central_pos_keine_randlage(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich vermeide Randlagen und Vororte, zentral ist Pflicht")[CENTRALITY] > 0

def test_central_pos_alles_erreichbar(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Sehr zentrale Lage im Stadtzentrum, zentrumsnah und urban wohnen")[CENTRALITY] > 0

def test_central_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich will im Stadtzentrum wohne, zentrale Lage isch mir sehr wichtig")[CENTRALITY] > 0

def test_central_neg_vorort(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ruhige Vorortlage bevorzugt, ausserhalb des Zentrums")[CENTRALITY] < 0

def test_central_neg_landlage(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Landlage oder Vorort stark bevorzugt, laendliche Umgebung")[CENTRALITY] < 0

def test_central_neg_agglomeration(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Agglomeration kein Problem, Stadtrand ist voellig in Ordnung")[CENTRALITY] < 0

def test_central_neg_vermeide_zentrum(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich vermeide das Stadtzentrum, zu laut und teuer")[CENTRALITY] < 0

def test_central_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ausserhalb dr Stadt wohne isch ok, ruhigi Vorortlage")[CENTRALITY] < 0

def test_central_ordering(augmenter: AnchorsAugmenter) -> None:
    central = _vec(augmenter, "Mitten im Stadtleben, zentrumsnahe Lage ein Muss")[CENTRALITY]
    suburb  = _vec(augmenter, "Laendliche Umgebung, weit ausserhalb des Zentrums")[CENTRALITY]
    assert central > suburb


# ── neighborhood_safety_score ─────────────────────────────────────────────────

def test_safety_pos_familienfreundlich(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Familienfreundliches sicheres Quartier fuer meine Kinder")[SAFETY] > 0

def test_safety_pos_schulweg(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Sicherer Schulweg, Kinder koennen alleine zur Schule gehen")[SAFETY] > 0

def test_safety_pos_kriminalarmes_quartier(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kriminalarmes Quartier, ich moechte mich nachts sicher fuehlen")[SAFETY] > 0

def test_safety_pos_spielplatz(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Spielplatz in der Naehe, Kinder koennen draussen spielen")[SAFETY] > 0

def test_safety_pos_ruhiges_familienquartier(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ruhiges Familienquartier, sichere Umgebung fuer die Kinder")[SAFETY] > 0

def test_safety_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Sicheri Umgaebig fuer d Chind, guets Familienquartier")[SAFETY] > 0

def test_safety_neg_sicherheit_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Sicherheit spielt keine grosse Rolle fuer mich")[SAFETY] < 0

def test_safety_neg_keine_prioritaet(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Sicherheit ist nicht meine Prioritaet, bin jung und allein")[SAFETY] < 0

def test_safety_neg_urbanes_quartier(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Urbanes lebhaftes Quartier auch nachts in Ordnung")[SAFETY] < 0

def test_safety_neg_kein_wert(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich lege keinen Wert auf ein ruhiges sicheres Quartier")[SAFETY] < 0

def test_safety_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Sicherheit isch mer noed so wichtig, Quartier muss noed sicher si")[SAFETY] < 0

def test_safety_ordering(augmenter: AnchorsAugmenter) -> None:
    safe   = _vec(augmenter, "Sicheres familienfreundliches Quartier, sicherer Schulweg")[SAFETY]
    unsafe = _vec(augmenter, "Sicherheit egal, urbanes Quartier auch nachts ok")[SAFETY]
    assert safe > unsafe


# ── proximity_to_desired_location_score ───────────────────────────────────────

def test_prox_pos_kurzer_arbeitsweg(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kurzer Arbeitsweg sehr wichtig, wenige Minuten zum Buero")[PROXIMITY] > 0

def test_prox_pos_uni_naehe(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Nah an der Universitaet, kurzer Weg zur ETH")[PROXIMITY] > 0

def test_prox_pos_eth_naehe(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "In der Naehe der ETH Zuerich wohnen, Pendelzeit minimal")[PROXIMITY] > 0

def test_prox_pos_hauptbahnhof(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kurze Distanz zum Hauptbahnhof, nah am Zielort")[PROXIMITY] > 0

def test_prox_pos_schulweg_kinder(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kurzer Schulweg fuer die Kinder, nahe bei Schule und Kita")[PROXIMITY] > 0

def test_prox_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Noed bi dr Arbet isch sehr wichtig, kurzer Weg zum Job")[PROXIMITY] > 0

def test_prox_neg_pendelweg_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Pendelweg spielt keine Rolle fuer mich")[PROXIMITY] < 0

def test_prox_neg_langer_weg_ok(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Langer Arbeitsweg ist kein Problem, ich hoere Podcasts")[PROXIMITY] < 0

def test_prox_neg_pendeln_ok(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich pendle gerne auch eine Stunde, Distanz egal")[PROXIMITY] < 0

def test_prox_neg_kein_zielort(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kein spezifischer Zielort noetig, arbeite flexibel")[PROXIMITY] < 0

def test_prox_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Pendeln isch kein Problem fuer mich, Arbeitsweg isch egal")[PROXIMITY] < 0

def test_prox_ordering(augmenter: AnchorsAugmenter) -> None:
    near = _vec(augmenter, "Naehe zum Arbeitsplatz ist ein Muss, Pendelzeit minimal")[PROXIMITY]
    far  = _vec(augmenter, "Distanz zum Arbeitsplatz egal, pendle gerne weit")[PROXIMITY]
    assert near > far


# ── air_quality_index ─────────────────────────────────────────────────────────

def test_air_pos_frische_luft(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Frische Luft sehr wichtig, weit weg von Abgasen")[AIR_QUALITY] > 0

def test_air_pos_keine_verschmutzung(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Keine Luftverschmutzung, saubere Luft in der Umgebung")[AIR_QUALITY] > 0

def test_air_pos_keine_hauptstrasse(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Nicht neben einer Hauptstrasse, keine Abgase vom Verkehr")[AIR_QUALITY] > 0

def test_air_pos_gute_luftqualitaet(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Gute Luftqualitaet im Quartier ist mir sehr wichtig")[AIR_QUALITY] > 0

def test_air_pos_abseits_verkehr(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Abseits von stark befahrenen Strassen, keine Abgase")[AIR_QUALITY] > 0

def test_air_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Frisch Luft isch mer sehr wichtig, kei Abgas vo dr Strass")[AIR_QUALITY] > 0

def test_air_neg_hauptstrasse_ok(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Naehe zur Hauptstrasse kein Problem fuer mich")[AIR_QUALITY] < 0

def test_air_neg_abgase_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Abgase stoeren mich nicht, Luftverschmutzung ist mir egal")[AIR_QUALITY] < 0

def test_air_neg_luftqualitaet_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Luftqualitaet spielt keine Rolle fuer mich")[AIR_QUALITY] < 0

def test_air_neg_hauptstrasse_direkt(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Direkt an der Hauptstrasse ist voellig ok, stoert mich nicht")[AIR_QUALITY] < 0

def test_air_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Luftqualitaet isch mer egal, Abgase stoere mich ueberhaupt noed")[AIR_QUALITY] < 0

def test_air_ordering(augmenter: AnchorsAugmenter) -> None:
    clean = _vec(augmenter, "Saubere Luft und wenig Abgase, Luftqualitaet sehr wichtig")[AIR_QUALITY]
    dirty = _vec(augmenter, "Luftqualitaet unwichtig, nah an der Hauptstrasse ist ok")[AIR_QUALITY]
    assert clean > dirty


# ── spaciousness_perception ───────────────────────────────────────────────────

def test_space_pos_grosse_zimmer(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Grosse Zimmer und hohe Decken, viel Platz gewuenscht")[SPACIOUSNESS] > 0

def test_space_pos_offener_grundriss(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Offener Grundriss, loftartige Wohnung mit viel Raum")[SPACIOUSNESS] > 0

def test_space_pos_viel_platz(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich brauche viel Platz, grosse Kueche und grosses Wohnzimmer")[SPACIOUSNESS] > 0

def test_space_pos_grosszuegig(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Grosszuegige Wohnflaeche, hohe Raeume und offenes Wohnen")[SPACIOUSNESS] > 0

def test_space_pos_stauraum(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Viel Stauraum und guter Grundriss, luftige Wohnung")[SPACIOUSNESS] > 0

def test_space_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Vill Platz zum Wohne isch wichtig, grossi Zimmer gewuenscht")[SPACIOUSNESS] > 0

def test_space_neg_studio_reicht(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kleines gemuetliches Studio reicht mir vollkommen")[SPACIOUSNESS] < 0

def test_space_neg_platz_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Platz ist nicht so wichtig, ich bin selten zuhause")[SPACIOUSNESS] < 0

def test_space_neg_kompakt(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kompaktes Apartment ist genug, ich brauche nicht viel Platz")[SPACIOUSNESS] < 0

def test_space_neg_kleine_wohnung(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kleine Wohnung ist voellig ok, Groesse spielt keine Rolle")[SPACIOUSNESS] < 0

def test_space_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kleini Wohnig reicht mir voellig, Platz isch noed so wichtig")[SPACIOUSNESS] < 0

def test_space_ordering(augmenter: AnchorsAugmenter) -> None:
    large = _vec(augmenter, "Sehr grosszuegige Wohnung, hohe Decken und viel Raum")[SPACIOUSNESS]
    small = _vec(augmenter, "Kompaktes Studio ist genug, ich brauche nicht viel Platz")[SPACIOUSNESS]
    assert large > small


# ── interior_modernity ────────────────────────────────────────────────────────

def test_modern_pos_renoviert(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Neu renovierte Wohnung mit moderner Kueche und Bad")[MODERNITY] > 0

def test_modern_pos_erstbezug(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Erstbezug nach Sanierung, Designerkueche und Fussbodenheizung")[MODERNITY] > 0

def test_modern_pos_vzug(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "V-Zug Geraete, Regendusche, Neubau oder frisch saniert")[MODERNITY] > 0

def test_modern_pos_hochwertig(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Hochwertige Innenausstattung, hochwertiger Parkett, Naturstein")[MODERNITY] > 0

def test_modern_pos_zeitgemaess(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Zeitgemaesse Einrichtung, alles neu und modern eingerichtet")[MODERNITY] > 0

def test_modern_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Moderni Chuchi und moderni Usstattung isch sehr wichtig")[MODERNITY] > 0

def test_modern_neg_altbau(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Altbau mit Charme wird bevorzugt, ich mag keine Neubauten")[MODERNITY] < 0

def test_modern_neg_rustikal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Rustikale traditionelle Ausstattung, Patina und Geschichte")[MODERNITY] < 0

def test_modern_neg_kein_modernes_design(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich mag kein steriles modernes Design, lieber Charakter")[MODERNITY] < 0

def test_modern_neg_historisch(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Historisches Gebaeude sehr schoen, Altbau mit Geschichte")[MODERNITY] < 0

def test_modern_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Moderni Wohnig isch noed mis Ding, lieber Altbau mit Charm")[MODERNITY] < 0

def test_modern_ordering(augmenter: AnchorsAugmenter) -> None:
    modern = _vec(augmenter, "Moderne hochwertige Ausstattung, alles neu renoviert")[MODERNITY]
    altbau = _vec(augmenter, "Altbau isch vill schoener, ich suche keinen Neubau")[MODERNITY]
    assert modern > altbau


# ── livability_score ──────────────────────────────────────────────────────────

def test_liva_pos_schoene_nachbarschaft(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Schoene Nachbarschaft, angenehmes Wohnumfeld sehr wichtig")[LIVABILITY] > 0

def test_liva_pos_hohe_lebensqualitaet(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Hohes Wohnqualitaetsniveau, gutes Quartier mit hoher Lebensqualitaet")[LIVABILITY] > 0

def test_liva_pos_tolle_nachbarschaft(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Tolle Nachbarschaft, gutes Klima und angenehme Atmosphaere")[LIVABILITY] > 0

def test_liva_pos_gutes_wohnumfeld(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich lege Wert auf ein gutes angenehmes Wohnumfeld")[LIVABILITY] > 0

def test_liva_pos_schoenes_quartier(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich moechte in einem schoenen Quartier mit gutem Klima leben")[LIVABILITY] > 0

def test_liva_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Hohe Wohnqualitaet isch mir wichtig, schoens Quartier mit gutem Wohnklima")[LIVABILITY] > 0

def test_liva_neg_hauptsache_guenstig(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Wohnqualitaet spielt keine Rolle, Hauptsache guenstig")[LIVABILITY] < 0

def test_liva_neg_nicht_waehlerisch(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich bin nicht waehlerisch beim Quartier, egal wo")[LIVABILITY] < 0

def test_liva_neg_wohnumfeld_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Wohnumfeld ist mir egal, Quartierqualitaet spielt keine Rolle")[LIVABILITY] < 0

def test_liva_neg_keine_ansprueche(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich stell keine hohen Ansprueche ans Wohnumfeld")[LIVABILITY] < 0

def test_liva_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Wohnqualitaet isch mer egal, Hauptsach guenstig und egal wo")[LIVABILITY] < 0

def test_liva_ordering(augmenter: AnchorsAugmenter) -> None:
    good  = _vec(augmenter, "Hohe Lebensqualitaet, schoenes angenehmes Wohnumfeld")[LIVABILITY]
    indif = _vec(augmenter, "Wohnqualitaet unwichtig, Hauptsache billig")[LIVABILITY]
    assert good > indif


# ── work_from_home_fitness ────────────────────────────────────────────────────

def test_wfh_pos_arbeitszimmer(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Separates Arbeitszimmer sehr wichtig, arbeite von zuhause")[WFH] > 0

def test_wfh_pos_schnelles_internet(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Schnelles Internet fuer Remote-Arbeit, Glasfaseranschluss")[WFH] > 0

def test_wfh_pos_homeoffice_ecke(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Homeoffice-Ecke oder eigener Bueroraum in der Wohnung")[WFH] > 0

def test_wfh_pos_videokonferenzen(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ruhige Wohnung fuer Videokonferenzen, gutes WLAN noetig")[WFH] > 0

def test_wfh_pos_remote_work(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Remote Work ist mein Alltag, ich bin auf schnelles Internet angewiesen")[WFH] > 0

def test_wfh_pos_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Home Office isch taeglich min Alltag, schnells Internet Muss")[WFH] > 0

def test_wfh_neg_gehe_ins_buero(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich gehe taeglich ins Buero, Home Office kein Thema")[WFH] < 0

def test_wfh_neg_kein_bedarf(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Kein Bedarf an Arbeitszimmer, arbeite nie von zuhause")[WFH] < 0

def test_wfh_neg_remote_work_egal(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Remote Work spielt keine Rolle, Homeoffice-Ausstattung unwichtig")[WFH] < 0

def test_wfh_neg_jeden_tag_buero(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Ich arbeite jeden Tag im Buero, kein Homeoffice noetig")[WFH] < 0

def test_wfh_neg_swiss_german(augmenter: AnchorsAugmenter) -> None:
    assert _vec(augmenter, "Home Office bruche ich ueberhaupt noed, gah taeglich is Buero")[WFH] < 0

def test_wfh_ordering(augmenter: AnchorsAugmenter) -> None:
    remote = _vec(augmenter, "Homeoffice-tauglich, Arbeitszimmer und schnelles Internet")[WFH]
    office = _vec(augmenter, "Kein Homeoffice, ich arbeite jeden Tag im Buero")[WFH]
    assert remote > office
