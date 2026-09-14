import json
import pandas as pd
import re
from pathlib import Path

def load_raw_events(path="data/raw_events.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    """Nettoie un texte : supprime markdown basique, espaces multiples, etc."""
    if not text:
        return ""
    text = re.sub(r'[*_#`]', '', text)          # markdown basique
    text = re.sub(r'\s+', ' ', text).strip()     # espaces multiples
    return text

def extract_event(e):
    """Extrait et structure les champs utiles d'un événement brut OpenAgenda."""
    location = e.get("location") or {}
    first_timing = e.get("firstTiming") or {}
    last_timing = e.get("lastTiming") or {}
    keywords = e.get("keywords") or []

    return {
        "uid": e.get("uid"),
        "title": clean_text(e.get("title")),
        "description": clean_text(e.get("description")),
        "long_description": clean_text(e.get("longDescription")),
        "keywords": ", ".join(keywords) if keywords else "",
        "city": location.get("adminLevel4") or location.get("city") or "",
        "department": location.get("adminLevel2") or "",
        "region": location.get("adminLevel1") or "",
        "postal_code": location.get("postalCode") or "",
        "venue_name": location.get("name") or "",
        "address": location.get("address") or "",
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "date_start": first_timing.get("begin"),
        "date_end": last_timing.get("end"),
        "slug": e.get("slug"),
        "url": f"https://openagenda.com/agenda-de-la-region-des-pays-de-la-loire/events/{e.get('slug')}" if e.get("slug") else "",
    }

def build_embedding_text(row):
    """Construit le texte qui sera vectorisé — condense l'info pertinente pour la recherche."""
    parts = [
        row["title"],
        row["description"],
        f"Lieu : {row['venue_name']}, {row['city']} ({row['department']})" if row["venue_name"] else f"Lieu : {row['city']} ({row['department']})",
        f"Mots-clés : {row['keywords']}" if row["keywords"] else "",
        f"Date : {row['date_start']}" if row["date_start"] else "",
    ]
    return " | ".join(p for p in parts if p)

def main():
    raw_events = load_raw_events()
    print(f"Événements bruts chargés : {len(raw_events)}")

    df = pd.DataFrame([extract_event(e) for e in raw_events])

    # Nettoyage : suppression des lignes sans titre ou sans date de début (inexploitables)
    before = len(df)
    df = df.dropna(subset=["title", "date_start"])
    df = df[df["title"].str.strip() != ""]
    print(f"Événements supprimés (titre/date manquant) : {before - len(df)}")

    # Conversion des dates en datetime pour usage ultérieur (tests, filtres)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)
    df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce", utc=True)

    # Suppression des doublons sur uid (sécurité, même si peu probable ici)
    before = len(df)
    df = df.drop_duplicates(subset="uid")
    print(f"Doublons supprimés : {before - len(df)}")

    # Exclusion des événements sans département (webinaires/événements en ligne
    # sans lieu physique) : hors périmètre pour un chatbot de recommandation
    # d'événements culturels géolocalisés.
    before = len(df)
    df = df[df["department"].notna() & (df["department"].str.strip() != "")]
    print(f"Événements en ligne/sans lieu exclus : {before - len(df)}")

    # Construction du texte destiné à la vectorisation
    df["embedding_text"] = df.apply(build_embedding_text, axis=1)

    # Sauvegarde
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/events_clean.csv", index=False)
    df.to_json("data/events_clean.json", orient="records", force_ascii=False, indent=2, date_format="iso")

    print(f"\n{len(df)} événements propres sauvegardés dans data/events_clean.csv et .json")
    print(f"\nAperçu :")
    print(df[["title", "city", "department", "date_start"]].head(5).to_string())

if __name__ == "__main__":
    main()
