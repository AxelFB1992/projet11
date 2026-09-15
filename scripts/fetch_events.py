import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAGENDA_API_KEY")
#On a décidé de remplacer
#AGENDA_UID = 16676449
#BASE_URL = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"

SOURCES = [
    {"uid": 16676449, "nom": "Agenda Région Pays de la Loire", "admin_level2": None},
    #{"uid": 28658827, "nom": "Le Kiosque Mayenne", "admin_level2": None},
    #{"uid": 56526489, "nom": "Les nuits de la Mayenne", "admin_level2": None},
    {"uid": 14115607, "nom": "Unidivers Oui sortir", "admin_level2": None},
]

def fetch_events_from_agenda(agenda_uid, admin_level2=None):
    """Récupère les événements à venir d'un agenda."""
    base_url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {
        "relative[]": "upcoming",
        "detailed": 1,
        "monolingual": "fr",
        "size": 100,
        "sort": "timingsWithFeatured.asc",
    }
    if admin_level2:
        params["adminLevel2[]"] = admin_level2

    all_events = []
    after = None
    while True:
        if after:
            params["after[]"] = after
        else:
            params.pop("after[]", None)

        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        all_events.extend(data.get("events", []))

        after = data.get("after")
        if not after:
            break

    return all_events

if __name__ == "__main__":
    all_events = []
    seen_uids = set()

    for source in SOURCES:
        events = fetch_events_from_agenda(source["uid"], source["admin_level2"])
        nouveaux = 0
        for e in events:
            if e["uid"] not in seen_uids:
                e["_source_agenda"] = source["nom"]  # traçabilité pour le rapport
                all_events.append(e)
                seen_uids.add(e["uid"])
                nouveaux += 1
        print(f"{source['nom']} : {len(events)} récupérés, {nouveaux} nouveaux (hors doublons)")

    output_path = "data/raw_events.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    print(f"\nTotal final : {len(all_events)} événements uniques sauvegardés dans {output_path}")
