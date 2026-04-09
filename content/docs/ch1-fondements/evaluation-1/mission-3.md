---
title: "Mission 3. Tokenizer Stress Test"
weight: 3
---

{{< katex />}}

# Évaluation 1 – Mission 3 : Tokenizer Stress Test & Analyse Adversariale
## Audit de la fragmentation et de l'intégrité sémantique (WordPiece vs BPE vs Byte-level)

Bonjour à toutes et à tous ! Dans cette mission, nous allons évaluer la robustesse algorithmique des schémas de tokenisation face à des données non conventionnelles (bruitées, techniques ou multilingues rares).

> [!IMPORTANT]
**Note technique :** 
La tokenisation est la première ligne de défense (ou de vulnérabilité) d'un LLM. Un tokeniseur qui fragmente excessivement le texte ou qui remplace des caractères par des tokens `[UNK]` (Unknown) sabote le travail des couches d'attention. Dans cette mission, nous testons la "résilience" des modèles face à la complexité du monde réel : mathématiques LaTeX, emojis composites et langues à faible ressources.

---

## 1. Préparation de l'environnement et des modèles
*Temps estimé : 2 minutes. Nous comparons trois générations de tokeniseurs aux philosophies distinctes.*

```python
# Installation des dépendances
# !pip install -q transformers torch pandas

from transformers import AutoTokenizer
import pandas as pd

# 1. Modèle historique (WordPiece - Petit vocabulaire)
tk_bert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# 2. Modèle standard (BPE - Moyen vocabulaire)
tk_gpt2 = AutoTokenizer.from_pretrained("gpt2")

# 3. Modèle moderne (BPE Byte-level - Grand vocabulaire + spécialisation)
tk_phi3 = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

tokenizers = {
    "BERT (WordPiece)": tk_bert,
    "GPT-2 (BPE)": tk_gpt2,
    "Phi-3 (Modern BPE)": tk_phi3
}
```

---

## 2. Définition du dataset "Adversarial"
Ce petit jeu de données est spécialement conçu pour piéger les algorithmes de découpage.

```python
adversarial_inputs = {
    "Math (LaTeX)": r"\frac{\partial^2 \psi}{\partial x^2} = \frac{1}{v^2} \frac{\partial^2 \psi}{\partial t^2}",
    "Langue Rare (Amharique)": "ሰላም ለሁላችሁ ይሁን",
    "Emoji ZWJ (Famille)": "👨‍👩‍👧‍👦", # ZWJ Sequence : Homme + Femme + Fille + Garçon
    "Code (Obfusqué)": "lambda x: [i**2 for i in x if i%2==0]"
}
```

---

## 3. Implémentation technique : Calcul du Ratio de Fragmentation
L'analyse se base sur le **Fragmentation Ratio** : $\text{Nombre de tokens} / \text{Nombre de mots (split espaces)}$. Un ratio élevé indique une perte d'efficacité sémantique.

```python
# --- CONFIGURATION DE LA MISSION (QUESTION CODE) ---
# TÂCHE : Itérez sur chaque texte et chaque tokenizer, puis calculez :
# 1. Le nombre de tokens obtenus.
# 2. Le Fragmentation Ratio (Nombre tokens / Nombre de mots originaux).
# 3. Le nombre de tokens [UNK] générés.

results = []

for category, text in adversarial_inputs.items():
    word_count = len(text.split()) # Nombre de mots originaux
    
    for name, tk in tokenizers.items():
        # Encodez le texte sans ajouter les tokens spéciaux (add_special_tokens=False)
        # ...
        
        # Récupérez la liste textuelle des tokens pour un aperçu avec convert_ids_to_tokens
        # ...
        
        # Comptez les tokens inconnus (attention : tous les modèles n'ont pas toujours de tk.unk_token)
        # ...
        
        # Enregistrez vos métriques dans la variable results
        pass

# Affichez les résultats (ex: via un pandas DataFrame) !
```

---

## 4. Analyse et Justifications (À remettre par l'étudiant)

**Résultats attendus :**
Vous devriez observer des comportements radicalement différents selon l'algorithme :
*   Le modèle WordPiece (BERT) va souvent **fragmenter excessivement** et potentiellement échouer sur certaines syntaxes comme les emojis ou certains opérateurs de code en produisant des tokens bizarres.
*   Le modèle BPE (GPT-2) utilise massivement le **fallback par octets** pour les langues rares, ce qui évite les `[UNK]` mais produit de longs vecteurs d'octets.
*   Le modèle moderne (Phi-3) devrait exceller sur le code et les emojis grâce à sa forte spécialisation, sa méthode Byte-level, et son très large vocabulaire.

> [!NOTE]
> * La solution complète avec le tableau des résultats Pandas est incluse dans le notebook Colab qui sera partagé avec le prof.

---

## Question Critique de Mission
**Question** : Qu'est-ce que la "Token Tax" Multilingue, et quel est l'avantage structurel du Byte-level BPE par rapport à une approche historique comme celle de BERT (WordPiece) ?

**Réponse attendue** : La "Token Tax" Multilingue désigne le fait que les langues non-latines ou à faibles ressources nécessitent beaucoup plus de tokens (et donc consomment plus de contexte et de VRAM) que l'anglais. Par exemple, le ratio de fragmentation est beaucoup plus lourd pour ces langues. L'avantage d'une approche Byte-level BPE (comme sur GPT-2 ou Phi-3) par rapport aux anciens modèles, c'est que le tokeniseur décompose n'importe quel caractère inconnu en ses octets UTF-8 sous-jacents : il n'utilise quasiment jamais de token `[UNK]` d'erreur. Cela garantit une préservation à 100% de l'information brute (Fallback), même si la représentation est fragmentée et coûteuse.

---

> [!TIP]
**Conclusion de l'Ingénieur** : Cette mission démontre formellement que le choix du tokeniseur est un compromis critique entre couverture globale, besoin de compression et limitation des ressources de calcul.
