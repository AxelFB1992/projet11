"""
Tests unitaires : vérifie que les événements intégrés dans la base correspondent bien au périmètre défini (région Pays de la Loire,
événements à venir dans une fenêtre de moins d'un an).
Ces tests sont donc à éxécuter avant la vectorisation pour savoir si les éléments stockés sont propres et pertinents pour l'index
Une fois que la vectorisation est faite, il n'y a plus de moyens de modifier les éléments qui sont déjà sous forme de vecteurs
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

#On récupère les élements via events_clean.csv (alors que pour la vectorisation, on les récupère au format .json)
DATA_PATH = Path("data/events_clean.csv")

# Les 5 départements constituant la région Pays de la Loire
DEPARTEMENTS_REGION = {
    "Loire-Atlantique", "Maine-et-Loire", "Mayenne", "Sarthe", "Vendée"
}


@pytest.fixture(scope="module")

#Permet de savoir le fichier renseigné dans le path (events_clean.csv) existe et si le jeu de données (df) n'est pas vide
def events_df():
    """Charge le jeu de données nettoyé une seule fois pour tous les tests."""
    assert DATA_PATH.exists(), f"Fichier introuvable : {DATA_PATH}. Lance d'abord fetch_events.py et preprocess_events.py."
    #On spécifie deux champs pour ne pas avoir à récuperer tout le jeu de données : ça nous dit si le df est vide ou pas
    df = pd.read_csv(DATA_PATH, parse_dates=["date_start", "date_end"])
    #Ce test est falcultatif car il sera réalisé par la prochaine fonction
    assert len(df) > 0, "Le jeu de données est vide."
    return df

#Il semblerait que la fonction prend en paramètre une autre fonction, qui lui retourne le dataframe
def test_data_not_empty(events_df):
    """Le jeu de données contient au moins un événement."""
    assert len(events_df) > 0

def test_no_missing_title(events_df):
    """Chaque événement a un titre non vide."""
    assert events_df["title"].notna().all()
    assert (events_df["title"].str.strip() != "").all()

#Pour savoir si une date de début est valide, on décide que tous les éléments de date_start ne doivent pas être null
def test_no_missing_date(events_df):
    """Chaque événement a une date de début valide."""
    assert events_df["date_start"].notna().all()


def test_no_duplicate_events(events_df):
    """Aucun événement n'est dupliqué (même uid)."""
    assert events_df["uid"].is_unique

#On force les évenements à tous être lié aux pays de la loire en l'écrivant en dur dans le test
def test_region_is_pays_de_la_loire(events_df):
    """Tous les événements sont bien situés en région Pays de la Loire."""
    regions_invalides = events_df[events_df["region"] != "Pays de la Loire"]
    #Il ne faut pas trouver d'évenements relatifs à des régions hors pays de la loire, sinon on les affiche clairement
    assert regions_invalides.empty, (
        f"{len(regions_invalides)} événement(s) hors région Pays de la Loire : "
        f"{regions_invalides['title'].tolist()}"
    )

#De même il ne faut pas trouver d'évenement qui ne concerne pas les départements listés ci-dessus (tous de pays de la loire)
def test_department_within_region(events_df):
    """Le département de chaque événement fait bien partie des 5 départements de la région."""
    departements_invalides = events_df[~events_df["department"].isin(DEPARTEMENTS_REGION)]
    assert departements_invalides.empty, (
        f"{len(departements_invalides)} événement(s) avec un département hors périmètre : "
        f"{departements_invalides[['title', 'department']].to_dict('records')}"
    )

#On fait un test pour savoir si les évenements ne sont pas plus vieux d'an et n'auront pas lieu dans plus d'un an
#Ce test est trop rigoureux car on a clairement spécifier dans fetch_events.py que l'on ne voulait que des évenements à venir (- 1an)
def test_events_are_upcoming_or_recent(events_df):
    """
    Chaque événement respecte la contrainte 'moins d'un an' :
    - pas plus d'un an dans le passé
    - pas plus d'un an dans le futur (fenêtre raisonnable pour un POC)
    """
    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)
    one_year_ahead = now + timedelta(days=365)

    # Normalise les dates en UTC si elles ne le sont pas déjà (au cas où)
    dates = events_df["date_start"]
    if dates.dt.tz is None:
        dates = dates.dt.tz_localize("UTC")

    trop_anciens = events_df[dates < one_year_ago]
    trop_lointains = events_df[dates > one_year_ahead]

    #Il ne faut ni qu'il y ait des dates trop anciennes...
    assert trop_anciens.empty, (
        f"{len(trop_anciens)} événement(s) datant de plus d'un an : "
        f"{trop_anciens['title'].tolist()}"
    )
    #... Ou des dates trop récentes, sinon on les affiche
    assert trop_lointains.empty, (
        f"{len(trop_lointains)} événement(s) prévus à plus d'un an : "
        f"{trop_lointains['title'].tolist()}"
    )

#Permet de savoir si le champs (colonne) test_embedding_text_not_empty n'est jamais vide pour toutes les lignes (all())
def test_embedding_text_not_empty(events_df):
    """Le texte préparé pour la vectorisation n'est jamais vide."""
    assert "embedding_text" in events_df.columns
    assert events_df["embedding_text"].notna().all()
    assert (events_df["embedding_text"].str.strip() != "").all()

#Test plus poussés qui permet d'éviter des grandes anomalies de géolocalisation, lorsque les coordonées sont renseignées
def test_coordinates_are_valid(events_df):
    """Les coordonnées GPS, quand présentes, sont dans des bornes plausibles pour la France."""
    coords = events_df.dropna(subset=["latitude", "longitude"])
    assert (coords["latitude"].between(41, 51)).all(), "Latitude hors bornes France métropolitaine"
    assert (coords["longitude"].between(-5, 10)).all(), "Longitude hors bornes France métropolitaine"
