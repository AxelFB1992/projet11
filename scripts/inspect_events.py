import json

with open("data/raw_events.json", "r", encoding="utf-8") as f:
    events = json.load(f)

print(f"Nombre d'événements : {len(events)}\n")
print("Clés disponibles sur le premier événement :")
print(sorted(events[0].keys()))

print("\n--- Aperçu des champs clés pour 3 événements ---")
for e in events[:3]:
    print(f"\nuid: {e.get('uid')}")
    print(f"title: {e.get('title')}")
    print(f"description: {e.get('description')}")
    print(f"location: {e.get('location')}")
    print(f"keywords: {e.get('keywords')}")
    print(f"firstTiming: {e.get('firstTiming')}")
    print(f"lastTiming: {e.get('lastTiming')}")
    print(f"slug: {e.get('slug')}")
