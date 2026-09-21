import requests
import os
from dotenv import load_dotenv
import json

#permet de charger les variables d'environnement présentes dans le fichier env.
load_dotenv()
API_KEY = os.getenv("OPENAGENDA_API_KEY")

"""les fonctions ci-dessous peuvent être appelé de manière unitaire : il n'y a pas de dépendances entre elles
Elles réalisent chacune des actions différentes allant de la simple recherche d'agenda à l'inspection dans le détail
de tous les évenements à venir pour un agenda"""

#Cette fonction permet de faire un recherche syntaxique sur les agendas présent dans openagenda
def search_agendas(query):
    #la base de l'url est toujours la même quel que soit le type de recherche que l'on va faire (agendas, evenements, lieux)
    url = "https://api.openagenda.com/v2/agendas"
    #En protocole http, le header permet de passer des informations non visibles dans l'url, ici la clé API pour requêter openAgendas
    headers = {"key": API_KEY}
    #Les paramètres vont être collés à l'url pour spécifer la demande. En l'occurence ici une recherche semantique avec une taille
    params = {"search": query, "size": 10}
    response = requests.get(url, headers=headers, params=params)
    #Permet d'attendre la réponse et de lever une erreur
    response.raise_for_status()
    #Retourne la réponse en format json, segmenté et balisé
    """La réponse initiale doit très probablement retourner du html sous format texte. On peut donc le parser avec json() pour obtenir
    une réponse réellement en json"""
    return response.json()

#Cette fonction permet d'obtenir des détails sur l'agenda dans sa globalité, pas sur les évenements auxquels il fait référence
def get_agenda_detail(uid):
    #Pour cette raison, on rajoute l'identifiant de l'agenda que l'on passe entre accolades
    url = f"https://api.openagenda.com/v2/agendas/{uid}"
    headers = {"key": API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

#Cette fonction permet de compter le nombre d'evenements que l'on a dans un agenda, utile pour savoir si il est pertinent ou pas
def count_events(agenda_uid, admin_level2=None):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    params = {"size": 1}
    #Si on a une description géographique de type 'département', alors elle sera utiliser pour filtrer les résultats de la recherche
    if admin_level2:
        params["adminLevel2[]"] = admin_level2
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    #Ici on va retourner non pas la réponse globale mais le nombre d'evenements retournées (soit le nombre d'elements dans la réponse)
    return response.json().get("total", 0)

def debug_dates(agenda_uid, nom):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    headers = {"key": API_KEY}
    #On trie les date de l'evenement le plus récent à celui le plus lointain
    params = {"size": 10, "sort": "timings.asc"}  # pas de filtre date
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"\n=== {nom} : dates brutes (sans filtre) ===")
    #Parmi tous les évenements
    for e in data.get("events", []):
        #On récupere le titre (si il y en a plusieurs on en recupère un : celui en français)
        title = e.get("title", {}).get("fr", "?") if isinstance(e.get("title"), dict) else e.get("title")
        #On récupère les dates (debut, fin, jours)
        timings = e.get("timings", [])
        #Et on ne selectionne que la date de début
        first = timings[0].get("begin") if timings else "?"
        #Que l'on affiche clairement
        print(f"  {first} | {title[:50]}")

#Cette fonction permet d'inspecter la structure d'un evenement en explorant le contenu entier (les balises) du premier évenement
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

#Permet d'inspecter plus en détail les différents champs liés au temps (timing ou date) afin de voir leur structure
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

#Permet de voir la répartition des évenements d'un agenda par département (adminLevel2)
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

#Le paramètre 'relative' permet de savoir si ce sont des évenements passés ou futurs que l'on recherche
#Par conséquent cette fonction permet de lister tous les évenements futur d'un agenda et d'indiquer la ville et le département
#Dans la limite de 50 caractère pour le titre et 20 poiur la ville
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
    #C'est le dernier agenda que l'on a inspecté avant de se focaliser dessus : les pays de la loire
    inspect_upcoming_region(16676449, "Agenda de la Région des Pays de la Loire")
    #inspect_upcoming_region(14115607, "Unidivers Oui sortir")
    inspect_upcoming_region(31651509, "Théâtre de Laval - CDN")
    inspect_upcoming_region(48454528, "Réseau des médiathèques & Archives du Mans")
