import json
import pandas as pd
import re
from pathlib import Path

#Cette fonction permet de récuperer le contenu de /raw_events.json et de le stocker en format .json (il est déjà parsé)
def load_raw_events(path="data/raw_events.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#Permet de nettoyer du texte en supprimant les caractères spéciaux (*_#) et en remplaçant les espaces multiples par un seul espace
#Cette fonction va être utilisé dans extract_event(e) pour nettoyer plusieurs champs de chaque évenement (title, description, ...)
def clean_text(text):
    """Nettoie un texte : supprime markdown basique, espaces multiples, etc."""
    if not text:
        return ""
    text = re.sub(r'[*_#`]', '', text)          # markdown basique
    text = re.sub(r'\s+', ' ', text).strip()     # espaces multiples
    return text

#Permet d'extraire 
def extract_event(e):
    """Extrait et structure les champs utiles d'un événement brut OpenAgenda."""
    #Certains champ sont complexe et contiennent des sous champs : on les recupère donc à l'avance en étape intermediaire
    location = e.get("location") or {}
    first_timing = e.get("firstTiming") or {}
    last_timing = e.get("lastTiming") or {}
    keywords = e.get("keywords") or []

    #Ces instructions permettent de récuperer les différents champs proprement et sous format texte, utilisable par la suite
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
        "source_agenda": e.get("_source_agenda", ""),
        "url": f"https://openagenda.com/agenda-de-la-region-des-pays-de-la-loire/events/{e.get('slug')}" if e.get("slug") else "",
    }

def build_embedding_text(row):
    """Construit le texte qui sera vectorisé — condense l'info pertinente pour la recherche."""
    # Récupère chaque ligne du data frame à vectoriser et le découpe en parties utiles pour l'index 
    # 4 informations sont pertinentes : le titre, la description (courte), le lieu (en pusieurs informations) et les mots-clés
    parts = [
        row["title"],
        row["description"],
        f"Lieu : {row['venue_name']}, {row['city']} ({row['department']})" if row["venue_name"] else f"Lieu : {row['city']} ({row['department']})",
        f"Mots-clés : {row['keywords']}" if row["keywords"] else "",
        #On retire cela car une date vectorisé ne veut rien dire : mieux vaut la garder comme métadonnées au moment de la vectorisation
        #f"Date : {row['date_start']}" if row["date_start"] else "",
    ]

    #L'élément date, au même titre que plusieurs autres éléments, sera conservé en tant que métadonnées lors de la vectorisation
    #En effet ce champs ne serait d'aucune utilité vectorisé, il va servir à cadrer la recherche de manière déterministe
    
    # Enfin, vous que l'on doit avoir une seule chaine de caractère pour la vectorisation, on concatène ces informations avec des '|'
    # Et on retourne le résultat
    return " | ".join(p for p in parts if p)

def main():
    #On récupere les évenement en format json (1 évenement = 1 élement json)
    raw_events = load_raw_events()
    #On compte le nombre d'évenement
    print(f"Événements bruts chargés : {len(raw_events)}")

    #Prend chaque élément json (document) et l'extrait proprement en utilisant extract_event(e) qui parse ses informations en texte
    df = pd.DataFrame([extract_event(e) for e in raw_events])

    #A partir de là on fonctionne directement avec le data frame que l'on a construit et on va faire du filtrage dessus

    # Nettoyage : suppression des lignes sans titre ou sans date de début (inexploitables)
    before = len(df)
    df = df.dropna(subset=["title", "date_start"])
    df = df[df["title"].str.strip() != ""]
    print(f"Événements supprimés (titre/date manquant) : {before - len(df)}")

    # Conversion des dates en datetime pour usage ultérieur (tests, filtres)
    #errors="coerce" permet de transformer le resultat de la conversion en NaT plutôt que de faire crasher tout le code si problème
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)
    df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce", utc=True)

    # Suppression des doublons sur uid (sécurité, même si peu probable ici)
    before = len(df)
    df = df.drop_duplicates(subset="uid")
    print(f"Doublons supprimés : {before - len(df)}")

    # Exclusion des événements sans département (webinaires/événements en ligne sans lieu physique) : 
    #hors périmètre pour un chatbot de recommandation d'événements culturels géolocalisés.
    #Cela va permet notamment de pouvoir utiliser ce critère comme métadonnés sur l'index pour filtrer les résultats
    before = len(df)
    df = df[df["department"].notna() & (df["department"].str.strip() != "")]
    print(f"Événements en ligne/sans lieu exclus : {before - len(df)}")

    # Exclusion des événements hors région Pays de la Loire (Unidivers est un agrégateur "Grand Ouest",
    # potentiellement plus large que le seul périmètre régional retenu pour ce POC).
    # En effet, on a rajouté Unidivers après en esperant que cet agenda ramène des résultats dans des départements avec peu
    # d'évenement comme la Mayenne ou la Sarthe, malheureusement cela n'a pas été le cas mais on l'a quand même garder
    before = len(df)
    df = df[df["region"] == "Pays de la Loire"]
    print(f"Événements hors région Pays de la Loire exclus : {before - len(df)}")


    # Exclusion des événements non culturels (orientation professionnelle, emploi, salons étudiants) hérités
    # de l'agenda institutionnel régional. Puls-Events est positionnée sur les événements culturels uniquement.
    # En effet, lors de nos recherches ultérieurs (avec l'index), les requêtes ramenaient souvent des évenements non-culturels
    NON_CULTURAL_PATTERNS = [
        r"\borientation\b",
        r"\bemploi\b",
        r"\bparcoursup\b",
        r"\brecrutement\b",
        r"\bjob\s?dating\b",
        r"\bcarrières?\b",
        r"\balternance\b",
        r"\bapprentissage\b",
        r"\binsertion professionnelle\b",
        r"salon de l'étudiant",
        r"salon des métiers",
        r"forum des métiers",
        r"forum.{0,15}métiers",
        r"\bstudyrama\b",
        r"big bang orientation",
    ]
    #On crée le regex avec le pattern précedent 
    exclusion_regex = re.compile("|".join(NON_CULTURAL_PATTERNS), re.IGNORECASE)
    #On récupère le texte contenant potentiellement les termes attestant d'un évenement non-culturel
    texte_complet = (df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["keywords"].fillna(""))
    #Et on en fait un masque qui référencent les champs des évenements non culturels
    mask_non_culturel = texte_complet.str.contains(exclusion_regex)
    #evenements_exclus = df[mask_non_culturel]

    before = len(df)
    #On liste les évenements correspondant à build_embedding_textbuild_embedding_textbuild_embedding_text]["title"].tolist()
    #Et on supprime ces évenements du dataframe initial, en prenant le soin de mesurer le nom d'envements exclus
    evenements_exclus = df[mask_non_culturel]["title"].tolist()
    df = df[~mask_non_culturel]
    print(f"Événements non culturels exclus : {before - len(df)}")
    for titre in evenements_exclus:
        print(f"    - {titre}")

    # Filtrage culturel renforcé pour la source "Département de la Vendée" uniquement.
    # Contrairement aux autres sources (agendas institutionnels culturels dédiés), celle-ci
    # est un agenda généraliste (conseils municipaux, vide-greniers, forums santé, sport...).
    # La liste noire ci-dessus ne suffit pas : on applique donc une liste BLANCHE de mots-clés
    # culturels, réservée aux événements de cette source, en plus du filtre déjà appliqué à tous.
    # Les deux filtres appliquées ne sont pas redondants mais compélmentaires (voir Notebook)
    CULTURAL_PATTERNS = [
        r"th[ée][aâ]tre", r"\bdanse\b", r"\bmusique\b", r"\bconcert\b",
        r"exposition", r"\bexpo\b", r"festival", r"spectacle",
        r"\bcirque\b", r"marionnette", r"op[ée]ra", r"cin[ée]ma",
        r"\bconte\b", r"po[ée]sie", r"litt[ée]rature", r"\bchorale\b",
        r"\bballet\b", r"m[ée]diath[èe]que", r"biblioth[èe]que",
        r"\bpeinture\b", r"\bsculpture\b", r"photographie", r"vernissage",
        r"patrimoine", r"art contemporain", r"atelier cr[ée]atif",
        r"\bgravure\b", r"illustration", r"arts plastiques", r"\bslam\b",
        r"spectacle vivant", r"\bhumour\b", r"conservatoire", r"\blecture\b",
    ]
    cultural_regex = re.compile("|".join(CULTURAL_PATTERNS), re.IGNORECASE)

    est_source_vendee = df["source_agenda"] == "Département de la Vendée"
    texte_complet_vendee = (df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["keywords"].fillna(""))
    mask_culturel_vendee = texte_complet_vendee.str.contains(cultural_regex)

    # On garde : tout événement qui NE vient PAS de cette source, OU qui en vient et matche la liste blanche
    before = len(df)
    df = df[~est_source_vendee | mask_culturel_vendee]
    print(f"Événements 'Département de la Vendée' exclus (hors périmètre culturel) : {before - len(df)}")

    # Construction du texte destiné à la vectorisation
    # Cet appel permet d'appliquer la fonction build_embedding_text soit aux lignes soit aux colonnes du dataframe
    # En l'occurrence ici, cela permet d'appliquer la fonction aux différences lignes (axis = 1) sans faire plusieurs appels
    # Cela va retourne une nouvelle colonne ' embedding_text' qui contiendra, pour chaque ligne un texte prêt à être vectorisé
    df["embedding_text"] = df.apply(build_embedding_text, axis=1)

    # On sauvegarde le résultat du dataframe en deux formats différents, au cas où : .csv et .json
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/events_clean.csv", index=False)
    df.to_json("data/events_clean.json", orient="records", force_ascii=False, indent=2, date_format="iso")

    print(f"\n{len(df)} événements propres sauvegardés dans data/events_clean.csv et .json")
    print(f"\nAperçu :")
    print(df[["title", "city", "department", "date_start"]].head(5).to_string())

    print(f"\nRépartition par département :")
    print(df["department"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
