import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()
API_KEY = os.getenv("OPENAGENDA_API_KEY")

def search_agendas(query):
    url = "https://api.openagenda.com/v2/agendas"
    headers = {"key": API_KEY}
    params = {"search": query, "size": 10}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def get_agenda_detail(uid):
    url = f"https://api.openagenda.com/v2/agendas/{uid}"
    headers = {"key": API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def count_events(agenda_uid, admin_level2=None):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {"size": 1}
    if admin_level2:
        params["adminLevel2[]"] = admin_level2
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("total", 0)

def debug_dates(agenda_uid, nom):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {"size": 10, "sort": "timings.asc"}  # pas de filtre date
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"\n=== {nom} : dates brutes (sans filtre) ===")
    for e in data.get("events", []):
        title = e.get("title", {}).get("fr", "?") if isinstance(e.get("title"), dict) else e.get("title")
        timings = e.get("timings", [])
        first = timings[0].get("begin") if timings else "?"
        print(f"  {first} | {title[:50]}")

def inspect_raw_structure(agenda_uid, nom):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {"size": 1, "detailed": 1}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"\n=== {nom} : structure complète du 1er événement ===")
    if data.get("events"):
        print(json.dumps(data["events"][0], indent=2, ensure_ascii=False)[:2000])
    else:
        print("Aucun événement retourné.")

def inspect_timings_field(agenda_uid, nom):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {"size": 3, "detailed": 1}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"\n=== {nom} : clés disponibles + timings ===")
    for e in data.get("events", []):
        title = e.get("title", {}).get("fr", "?") if isinstance(e.get("title"), dict) else e.get("title")
        print(f"\n--- {title[:60]} ---")
        print("Clés top-niveau contenant 'tim' ou 'date':",
              [k for k in e.keys() if "tim" in k.lower() or "date" in k.lower()])
        print("Contenu de 'timings':", json.dumps(e.get("timings"), ensure_ascii=False)[:500])

def inspect_locations(agenda_uid, nom, sample_size=50):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {"size": sample_size, "detailed": 1}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"\n=== {nom} : répartition géographique sur {len(data.get('events', []))} événements ===")
    from collections import Counter
    depts = Counter()
    for e in data.get("events", []):
        loc = e.get("location") or {}
        dept = loc.get("adminLevel2", "?")
        depts[dept] += 1
    for dept, count in depts.most_common(15):
        print(f"  {dept} : {count}")

def inspect_upcoming_region(agenda_uid, nom):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {
        "relative[]": "upcoming",
        "size": 50,
        "detailed": 1,
        "sort": "timingsWithFeatured.asc",
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"\n=== {nom} : {data.get('total')} événements à venir (toute la région) ===")
    for e in data.get("events", []):
        title = e.get("title", {}).get("fr", "?") if isinstance(e.get("title"), dict) else e.get("title")
        city = (e.get("location") or {}).get("city", "?")
        dept = (e.get("location") or {}).get("adminLevel2", "?")
        print(f"  {title[:50]:<50} | {city:<20} | {dept}")


if __name__ == "__main__":
    inspect_upcoming_region(16676449, "Agenda de la Région des Pays de la Loire")
