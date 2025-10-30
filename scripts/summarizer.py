import sys
import math
import warnings

# Silence noisy library warnings (urllib3 OpenSSL, transformers tokenizer notices)
# MUST be declared before importing transformers or any library that imports urllib3
warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 v2 only supports OpenSSL.*",
)
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

from transformers import pipeline
from transformers.utils import logging as hf_logging

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers",
)
hf_logging.set_verbosity_error()

# Chargement du modèle de résumé
summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    tokenizer="csebuetnlp/mT5_multilingual_XLSum",
    use_fast=False,
)

# Longueur max des tokens pour le modèle
CHUNK_SIZE = 300  # ~400 tokens ≈ 700-900 caractères FR
SUMMARY_MAX = 80
SUMMARY_MIN = 40


def decouper_texte(texte):
    """
    Découpe le texte en morceaux adaptés au modèle.
    On découpe par paragraphes puis on ajuste si trop long.
    """
    paragraphs = texte.split("\n")
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < 900:  # limite caractères approximative
            current += p + "\n"
        else:
            chunks.append(current)
            current = p + "\n"

    if current:
        chunks.append(current)

    return chunks


def resumer_chunk(chunk):
    resultat = summarizer(
        chunk,
        max_length=SUMMARY_MAX,
        min_length=SUMMARY_MIN,
        do_sample=False
    )
    return resultat[0]["summary_text"]


def resumer_document(texte):
    chunks = decouper_texte(texte)
    print(f"Découpage en {len(chunks)} partie(s)...")

    résumés_partiels = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Résumé de la partie {i}/{len(chunks)}...")
        résumé = summarizer(
            chunk,
            max_length=80,  # augmente la taille autorisée
            min_length=40,   # assure un résumé détaillé
            do_sample=False
        )[0]["summary_text"]

        résumés_partiels.append(f"### Partie {i}\n{résumé}\n")

    # Au lieu de re-résumer : on assemble tout proprement ✅
    return "\n".join(résumés_partiels)



def charger_fichier(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def sauvegarder_fichier(path, texte):
    with open(path, "w", encoding="utf-8") as f:
        f.write(texte)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python resume_long.py input.txt output.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    texte_source = charger_fichier(input_file)
    résumé_final = resumer_document(texte_source)
    sauvegarder_fichier(output_file, résumé_final)

    print(f"✅ Résumé sauvegardé dans : {output_file}")
