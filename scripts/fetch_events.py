import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAGENDA_API_KEY")
AGENDA_UID = 16676449
BASE_URL = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"

def fetch_all_upcoming_events():
    """Récupère tous les événements à venir de la région Pays de la Loire."""
    headers = {"key": API_KEY}
    params = {
        "relative[]": "upcoming",
        "detailed": 1,
        "monolingual": "fr",
        "size": 100,
        "sort": "timingsWithFeatured.asc",
    }

    all_events = []
    after = None

    while True:
        if after:
            params["after[]"] = after
        else:
            params.pop("after[]", None)

        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        events = data.get("events", [])
        all_events.extend(events)
        print(f"Récupérés : {len(all_events)} / {data.get('total')}")

        after = data.get("after")
        if not after:
            break

    return all_events

if __name__ == "__main__":
    events = fetch_all_upcoming_events()
    output_path = "data/raw_events.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    print(f"\n{len(events)} événements sauvegardés dans {output_path}")
